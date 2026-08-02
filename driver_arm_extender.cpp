/*
 * ArmExtender SteamVR Driver - Virtual Controller Mirror
 *
 * Creates two virtual controller devices that mirror all input from
 * your real controllers but with extended arm poses.
 *
 * How it works:
 *   1. On init, registers Left and Right virtual controllers
 *   2. Every frame, reads real controller poses + button state
 *   3. Applies shoulder-pivot arm extension to positions
 *   4. Feeds modified poses + exact button state to virtual controllers
 *   5. Games see the virtual controllers instead of real ones
 *
 * Install: SteamVR/drivers/arm_extender/bin/win64/driver_arm_extender.dll
 * Config:  %APPDATA%\ArmExtender\arm_extender.cfg  (hot-reloaded every 2s)
 */

#include <openvr_driver.h>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

template<typename T>
static T clamp_val(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Vec3 {
    double x, y, z;
    Vec3(double x=0,double y=0,double z=0):x(x),y(y),z(z){}
    Vec3 operator+(const Vec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x,y-o.y,z-o.z}; }
    Vec3 operator*(double s)       const { return {x*s,y*s,z*s}; }
    double dot(const Vec3& o)      const { return x*o.x+y*o.y+z*o.z; }
    double length()                const { return std::sqrt(dot(*this)); }
    Vec3 normalized() const {
        double l=length(); return l>1e-8?(*this)*(1.0/l):Vec3{};
    }
};

static Vec3 mat_pos(const vr::HmdMatrix34_t& m) {
    return {m.m[0][3],m.m[1][3],m.m[2][3]};
}
static Vec3 mat_rotate(const vr::HmdMatrix34_t& m, const Vec3& v) {
    return {
        m.m[0][0]*v.x+m.m[0][1]*v.y+m.m[0][2]*v.z,
        m.m[1][0]*v.x+m.m[1][1]*v.y+m.m[1][2]*v.z,
        m.m[2][0]*v.x+m.m[2][1]*v.y+m.m[2][2]*v.z
    };
}

// Matrix -> quaternion
static void mat_to_quat(const vr::HmdMatrix34_t& m,
                         double& qw, double& qx, double& qy, double& qz) {
    double trace = m.m[0][0]+m.m[1][1]+m.m[2][2];
    if (trace > 0) {
        double s = 0.5/std::sqrt(trace+1.0);
        qw = 0.25/s;
        qx = (m.m[2][1]-m.m[1][2])*s;
        qy = (m.m[0][2]-m.m[2][0])*s;
        qz = (m.m[1][0]-m.m[0][1])*s;
    } else if (m.m[0][0]>m.m[1][1] && m.m[0][0]>m.m[2][2]) {
        double s = 2.0*std::sqrt(1.0+m.m[0][0]-m.m[1][1]-m.m[2][2]);
        qw = (m.m[2][1]-m.m[1][2])/s;
        qx = 0.25*s;
        qy = (m.m[0][1]+m.m[1][0])/s;
        qz = (m.m[0][2]+m.m[2][0])/s;
    } else if (m.m[1][1]>m.m[2][2]) {
        double s = 2.0*std::sqrt(1.0+m.m[1][1]-m.m[0][0]-m.m[2][2]);
        qw = (m.m[0][2]-m.m[2][0])/s;
        qx = (m.m[0][1]+m.m[1][0])/s;
        qy = 0.25*s;
        qz = (m.m[1][2]+m.m[2][1])/s;
    } else {
        double s = 2.0*std::sqrt(1.0+m.m[2][2]-m.m[0][0]-m.m[1][1]);
        qw = (m.m[1][0]-m.m[0][1])/s;
        qx = (m.m[0][2]+m.m[2][0])/s;
        qy = (m.m[1][2]+m.m[2][1])/s;
        qz = 0.25*s;
    }
}

// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------

struct Config {
    float extensionFactor  = 1.3f;
    float shoulderX        = 0.18f;
    float shoulderY        = -0.20f;
    float shoulderZ        = 0.08f;
    float smoothing        = 0.05f;
    float minArmLen        = 0.10f;
    bool  enableLeft       = true;
    bool  enableRight      = true;

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string k,eq,v;
            if (!(ss>>k>>eq>>v)||eq!="=") continue;
            try {
                float fv = std::stof(v);
                if (k=="extensionFactor") extensionFactor=fv;
                if (k=="shoulderOffsetX") shoulderX=fv;
                if (k=="shoulderOffsetY") shoulderY=fv;
                if (k=="shoulderOffsetZ") shoulderZ=fv;
                if (k=="smoothingFactor") smoothing=fv;
                if (k=="minArmLength")    minArmLen=fv;
                if (k=="enableLeft")      enableLeft=(fv!=0);
                if (k=="enableRight")     enableRight=(fv!=0);
            } catch(...) {}
        }
    }
};

// --------------------------------------------------------------------------
// Smoother
// --------------------------------------------------------------------------

struct Smoother {
    Vec3 pos{};
    bool init=false;
    Vec3 smooth(const Vec3& t, float a) {
        if (!init){pos=t;init=true;}
        pos.x+=a*(t.x-pos.x);
        pos.y+=a*(t.y-pos.y);
        pos.z+=a*(t.z-pos.z);
        return pos;
    }
    void reset(){init=false;}
};

// --------------------------------------------------------------------------
// Virtual Controller Device
// --------------------------------------------------------------------------

class CVirtualController : public vr::ITrackedDeviceServerDriver {
public:
    CVirtualController(bool isRight, Config* cfg)
        : m_isRight(isRight), m_cfg(cfg) {
        m_pose.poseIsValid = false;
        m_pose.deviceIsConnected = false;
        m_pose.result = vr::TrackingResult_Uninitialized;
        m_pose.qWorldFromDriverRotation.w = 1;
        m_pose.qWorldFromDriverRotation.x = 0;
        m_pose.qWorldFromDriverRotation.y = 0;
        m_pose.qWorldFromDriverRotation.z = 0;
        m_pose.qDriverFromHeadRotation.w = 1;
        m_pose.qDriverFromHeadRotation.x = 0;
        m_pose.qDriverFromHeadRotation.y = 0;
        m_pose.qDriverFromHeadRotation.z = 0;
        m_pose.qRotation.w = 1;
        std::memset(m_pose.vecPosition, 0, sizeof(m_pose.vecPosition));
        std::memset(m_pose.vecVelocity, 0, sizeof(m_pose.vecVelocity));
        std::memset(m_pose.vecAcceleration, 0, sizeof(m_pose.vecAcceleration));
        std::memset(m_pose.vecAngularVelocity, 0, sizeof(m_pose.vecAngularVelocity));
        std::memset(m_pose.vecAngularAcceleration, 0, sizeof(m_pose.vecAngularAcceleration));
    }

    vr::EVRInitError Activate(uint32_t id) override {
        m_id = id;
        m_props = vr::VRProperties()->TrackedDeviceToPropertyContainer(id);

        // Identify as a controller
        vr::VRProperties()->SetStringProperty(m_props,
            vr::Prop_TrackingSystemName_String, "arm_extender");
        vr::VRProperties()->SetStringProperty(m_props,
            vr::Prop_ModelNumber_String,
            m_isRight ? "ArmExtender Right" : "ArmExtender Left");
        vr::VRProperties()->SetStringProperty(m_props,
            vr::Prop_SerialNumber_String,
            m_isRight ? "AE_RIGHT_001" : "AE_LEFT_001");
        vr::VRProperties()->SetStringProperty(m_props,
            vr::Prop_ManufacturerName_String, "ArmExtender");
        vr::VRProperties()->SetInt32Property(m_props,
            vr::Prop_DeviceClass_Int32, (int32_t)vr::TrackedDeviceClass_Controller);
        vr::VRProperties()->SetInt32Property(m_props,
            vr::Prop_ControllerRoleHint_Int32,
            m_isRight ? (int32_t)vr::TrackedControllerRole_RightHand
                      : (int32_t)vr::TrackedControllerRole_LeftHand);

        // Set up input components
        vr::VRDriverInput()->CreateBooleanComponent(m_props, "/input/trigger/click", &m_triggerClick);
        vr::VRDriverInput()->CreateScalarComponent(m_props, "/input/trigger/value",
            &m_triggerValue, vr::VRScalarType_Absolute, vr::VRScalarUnits_NormalizedOneSided);
        vr::VRDriverInput()->CreateBooleanComponent(m_props, "/input/grip/click", &m_gripClick);
        vr::VRDriverInput()->CreateScalarComponent(m_props, "/input/grip/value",
            &m_gripValue, vr::VRScalarType_Absolute, vr::VRScalarUnits_NormalizedOneSided);
        vr::VRDriverInput()->CreateBooleanComponent(m_props, "/input/system/click", &m_systemClick);
        vr::VRDriverInput()->CreateBooleanComponent(m_props, "/input/application_menu/click", &m_menuClick);
        vr::VRDriverInput()->CreateBooleanComponent(m_props, "/input/trackpad/click", &m_padClick);
        vr::VRDriverInput()->CreateBooleanComponent(m_props, "/input/trackpad/touch", &m_padTouch);
        vr::VRDriverInput()->CreateScalarComponent(m_props, "/input/trackpad/x",
            &m_padX, vr::VRScalarType_Absolute, vr::VRScalarUnits_NormalizedTwoSided);
        vr::VRDriverInput()->CreateScalarComponent(m_props, "/input/trackpad/y",
            &m_padY, vr::VRScalarType_Absolute, vr::VRScalarUnits_NormalizedTwoSided);
        vr::VRDriverInput()->CreateHapticComponent(m_props, "/output/haptic", &m_haptic);

        return vr::VRInitError_None;
    }

    void Deactivate() override { m_id = vr::k_unTrackedDeviceIndexInvalid; }
    void EnterStandby() override {}
    void* GetComponent(const char*) override { return nullptr; }
    void DebugRequest(const char*, char* buf, uint32_t sz) override { if(sz) buf[0]=0; }
    vr::DriverPose_t GetPose() override { return m_pose; }

    // Called every frame with real controller data
    void UpdateFromReal(
        const vr::TrackedDevicePose_t& realPose,
        const vr::VRControllerState_t& state,
        const vr::HmdMatrix34_t& hmdMat,
        bool hmdValid)
    {
        // --- Build extended pose ---
        if (realPose.bPoseIsValid && hmdValid) {
            float sx = m_isRight ? m_cfg->shoulderX : -m_cfg->shoulderX;
            Vec3 shoulderLocal{sx, m_cfg->shoulderY, m_cfg->shoulderZ};
            Vec3 hmdPos = mat_pos(hmdMat);
            Vec3 shoulder = hmdPos + mat_rotate(hmdMat, shoulderLocal);

            Vec3 ctrl{
                realPose.mDeviceToAbsoluteTracking.m[0][3],
                realPose.mDeviceToAbsoluteTracking.m[1][3],
                realPose.mDeviceToAbsoluteTracking.m[2][3]
            };

            Vec3 arm = ctrl - shoulder;
            double armLen = arm.length();

            Vec3 finalPos = ctrl;
            if (armLen >= m_cfg->minArmLen) {
                Vec3 extended = shoulder + arm.normalized() * (armLen * m_cfg->extensionFactor);
                float alpha = clamp_val(1.0f - m_cfg->smoothing, 0.01f, 1.0f);
                finalPos = m_smoother.smooth(extended, alpha);
            } else {
                m_smoother.reset();
            }

            m_pose.vecPosition[0] = finalPos.x;
            m_pose.vecPosition[1] = finalPos.y;
            m_pose.vecPosition[2] = finalPos.z;
            m_pose.vecVelocity[0] = realPose.vVelocity.v[0] * m_cfg->extensionFactor;
            m_pose.vecVelocity[1] = realPose.vVelocity.v[1] * m_cfg->extensionFactor;
            m_pose.vecVelocity[2] = realPose.vVelocity.v[2] * m_cfg->extensionFactor;
            m_pose.vecAngularVelocity[0] = realPose.vAngularVelocity.v[0];
            m_pose.vecAngularVelocity[1] = realPose.vAngularVelocity.v[1];
            m_pose.vecAngularVelocity[2] = realPose.vAngularVelocity.v[2];

            double qw,qx,qy,qz;
            mat_to_quat(realPose.mDeviceToAbsoluteTracking, qw,qx,qy,qz);
            m_pose.qRotation.w=qw; m_pose.qRotation.x=qx;
            m_pose.qRotation.y=qy; m_pose.qRotation.z=qz;

            m_pose.poseIsValid = true;
            m_pose.result = vr::TrackingResult_Running_OK;
        } else {
            m_smoother.reset();
            m_pose.poseIsValid = false;
            m_pose.result = vr::TrackingResult_Running_OutOfRange;
        }

        m_pose.deviceIsConnected = realPose.bDeviceIsConnected;

        // Submit pose
        if (m_id != vr::k_unTrackedDeviceIndexInvalid)
            vr::VRServerDriverHost()->TrackedDevicePoseUpdated(m_id, m_pose, sizeof(m_pose));

        // Mirror button state
        uint64_t pressed = state.ulButtonPressed;
        uint64_t touched = state.ulButtonTouched;

        auto btn = [](uint64_t mask, uint32_t bit) -> bool {
            return (mask & vr::ButtonMaskFromId((vr::EVRButtonId)bit)) != 0;
        };

        vr::VRDriverInput()->UpdateBooleanComponent(m_triggerClick,
            btn(pressed, vr::k_EButton_SteamVR_Trigger), 0);
        vr::VRDriverInput()->UpdateScalarComponent(m_triggerValue,
            state.rAxis[1].x, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(m_gripClick,
            btn(pressed, vr::k_EButton_Grip), 0);
        vr::VRDriverInput()->UpdateScalarComponent(m_gripValue,
            btn(pressed, vr::k_EButton_Grip) ? 1.0f : 0.0f, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(m_systemClick,
            btn(pressed, vr::k_EButton_System), 0);
        vr::VRDriverInput()->UpdateBooleanComponent(m_menuClick,
            btn(pressed, vr::k_EButton_ApplicationMenu), 0);
        vr::VRDriverInput()->UpdateBooleanComponent(m_padClick,
            btn(pressed, vr::k_EButton_SteamVR_Touchpad), 0);
        vr::VRDriverInput()->UpdateBooleanComponent(m_padTouch,
            btn(touched, vr::k_EButton_SteamVR_Touchpad), 0);
        vr::VRDriverInput()->UpdateScalarComponent(m_padX, state.rAxis[0].x, 0);
        vr::VRDriverInput()->UpdateScalarComponent(m_padY, state.rAxis[0].y, 0);
    }

    uint32_t GetId() const { return m_id; }

private:
    bool     m_isRight;
    Config*  m_cfg;
    uint32_t m_id = vr::k_unTrackedDeviceIndexInvalid;
    vr::PropertyContainerHandle_t m_props = vr::k_ulInvalidPropertyContainer;
    vr::DriverPose_t m_pose{};
    Smoother m_smoother;

    vr::VRInputComponentHandle_t
        m_triggerClick=0, m_triggerValue=0,
        m_gripClick=0,    m_gripValue=0,
        m_systemClick=0,  m_menuClick=0,
        m_padClick=0,     m_padTouch=0,
        m_padX=0,         m_padY=0,
        m_haptic=0;
};

// --------------------------------------------------------------------------
// Driver Provider
// --------------------------------------------------------------------------

class CArmExtenderProvider : public vr::IServerTrackedDeviceProvider {
public:
    vr::EVRInitError Init(vr::IVRDriverContext* ctx) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(ctx);

        // Load config
        const char* appData = std::getenv("APPDATA");
        if (appData) {
            m_cfgPath = std::string(appData) + "\\ArmExtender\\arm_extender.cfg";
            m_cfg.load(m_cfgPath);
        }

        // Register virtual controllers
        m_left  = new CVirtualController(false, &m_cfg);
        m_right = new CVirtualController(true,  &m_cfg);
        vr::VRServerDriverHost()->TrackedDeviceAdded("AE_LEFT_001",
            vr::TrackedDeviceClass_Controller, m_left);
        vr::VRServerDriverHost()->TrackedDeviceAdded("AE_RIGHT_001",
            vr::TrackedDeviceClass_Controller, m_right);

        // Config hot-reload thread
        m_running = true;
        m_thread = std::thread([this]() {
            while (m_running) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                if (!m_cfgPath.empty()) m_cfg.load(m_cfgPath);
            }
        });

        return vr::VRInitError_None;
    }

    void Cleanup() override {
        m_running = false;
        if (m_thread.joinable()) m_thread.join();
        delete m_left;
        delete m_right;
        VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }

    const char* const* GetInterfaceVersions() override { return vr::k_InterfaceVersions; }

    void RunFrame() override {
        // Get all poses
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        vr::VRServerDriverHost()->GetRawTrackedDevicePoses(0.f, poses, vr::k_unMaxTrackedDeviceCount);

        const auto& hmdPose = poses[vr::k_unTrackedDeviceIndex_Hmd];
        bool hmdValid = hmdPose.bPoseIsValid;
        const vr::HmdMatrix34_t& hmdMat = hmdPose.mDeviceToAbsoluteTracking;

        // Find real left and right controllers
        for (uint32_t i = 1; i < vr::k_unMaxTrackedDeviceCount; ++i) {
            if (!poses[i].bDeviceIsConnected) continue;

            // Skip our own virtual controllers
            if (m_left  && i == m_left->GetId())  continue;
            if (m_right && i == m_right->GetId()) continue;

            vr::ETrackedPropertyError err;
            int32_t cls = (int32_t)vr::VRProperties()->GetInt32Property(
                vr::VRProperties()->TrackedDeviceToPropertyContainer(i),
                vr::Prop_DeviceClass_Int32, &err);
            if (cls != (int32_t)vr::TrackedDeviceClass_Controller) continue;

            int32_t role = (int32_t)vr::VRProperties()->GetInt32Property(
                vr::VRProperties()->TrackedDeviceToPropertyContainer(i),
                vr::Prop_ControllerRoleHint_Int32, &err);

            vr::VRControllerState_t state{};
            vr::VRServerDriverHost()->GetRawTrackedDevicePoses(0.f, poses, vr::k_unMaxTrackedDeviceCount);

            if (role == (int32_t)vr::TrackedControllerRole_LeftHand && m_cfg.enableLeft && m_left)
                m_left->UpdateFromReal(poses[i], state, hmdMat, hmdValid);
            else if (role == (int32_t)vr::TrackedControllerRole_RightHand && m_cfg.enableRight && m_right)
                m_right->UpdateFromReal(poses[i], state, hmdMat, hmdValid);
        }
    }

    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

private:
    Config   m_cfg;
    std::string m_cfgPath;
    CVirtualController* m_left  = nullptr;
    CVirtualController* m_right = nullptr;
    std::thread         m_thread;
    std::atomic<bool>   m_running{false};
};

static CArmExtenderProvider g_provider;

extern "C" __declspec(dllexport)
void* HmdDriverFactory(const char* name, int* err) {
    if (std::string(name) == vr::IServerTrackedDeviceProvider_Version)
        return &g_provider;
    if (err) *err = vr::VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
