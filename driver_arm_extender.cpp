/*
 * ArmExtender SteamVR Driver
 * Biomechanically correct arm extension via shoulder-pivot scaling.
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

// --------------------------------------------------------------------------
// Math helpers
// --------------------------------------------------------------------------

struct Vec3 {
    double x, y, z;
    Vec3(double x=0,double y=0,double z=0): x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)       const { return {x*s,   y*s,   z*s};   }
    double dot(const Vec3& o)      const { return x*o.x + y*o.y + z*o.z; }
    double length()                const { return std::sqrt(dot(*this));   }
    Vec3 normalized() const {
        double l = length();
        return l > 1e-8 ? (*this)*(1.0/l) : Vec3{};
    }
};

template<typename T>
static T clamp_val(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

static Vec3 posFromMatrix(const vr::HmdMatrix34_t& m) {
    return {m.m[0][3], m.m[1][3], m.m[2][3]};
}

static Vec3 rotateByMatrix(const vr::HmdMatrix34_t& m, const Vec3& v) {
    return {
        m.m[0][0]*v.x + m.m[0][1]*v.y + m.m[0][2]*v.z,
        m.m[1][0]*v.x + m.m[1][1]*v.y + m.m[1][2]*v.z,
        m.m[2][0]*v.x + m.m[2][1]*v.y + m.m[2][2]*v.z
    };
}

// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------

struct ArmExtenderSettings {
    float extensionFactor   = 1.3f;
    float shoulderOffsetX   = 0.18f;
    float shoulderOffsetY   = -0.20f;
    float shoulderOffsetZ   = 0.08f;
    float smoothingFactor   = 0.05f;
    float minArmLength      = 0.10f;
    bool enableLeft         = true;
    bool enableRight        = true;

    void loadFromFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string key, eq, val;
            if (!(ss >> key >> eq >> val) || eq != "=") continue;
            try {
                float fv = std::stof(val);
                if (key == "extensionFactor")  extensionFactor  = fv;
                if (key == "shoulderOffsetX")  shoulderOffsetX  = fv;
                if (key == "shoulderOffsetY")  shoulderOffsetY  = fv;
                if (key == "shoulderOffsetZ")  shoulderOffsetZ  = fv;
                if (key == "smoothingFactor")  smoothingFactor  = fv;
                if (key == "minArmLength")     minArmLength     = fv;
                if (key == "enableLeft")       enableLeft       = (fv != 0.f);
                if (key == "enableRight")      enableRight      = (fv != 0.f);
            } catch (...) {}
        }
    }
};

// --------------------------------------------------------------------------
// Pose smoother
// --------------------------------------------------------------------------

struct PoseSmoother {
    Vec3 smoothPos{};
    bool initialised = false;

    Vec3 smooth(const Vec3& target, float alpha) {
        if (!initialised) { smoothPos = target; initialised = true; }
        smoothPos.x += alpha * (target.x - smoothPos.x);
        smoothPos.y += alpha * (target.y - smoothPos.y);
        smoothPos.z += alpha * (target.z - smoothPos.z);
        return smoothPos;
    }
    void reset() { initialised = false; }
};

// --------------------------------------------------------------------------
// Core arm extension — works on DriverPose_t
// --------------------------------------------------------------------------

static void extendArm(
    const vr::HmdMatrix34_t&   hmdMat,
    bool                        hmdValid,
    vr::DriverPose_t&           pose,
    bool                        isRightHand,
    const ArmExtenderSettings&  settings,
    PoseSmoother&               smoother)
{
    if (!hmdValid || pose.poseIsValid == false) {
        smoother.reset();
        return;
    }

    // Controller world position from DriverPose vecPosition (device-to-absolute)
    // vecPosition is in driver space; we need to work in the same space as HMD
    Vec3 ctrlPos{ pose.vecPosition[0], pose.vecPosition[1], pose.vecPosition[2] };

    // Shoulder in world space (estimated from HMD)
    float sx = isRightHand ? settings.shoulderOffsetX : -settings.shoulderOffsetX;
    Vec3 shoulderLocal{ sx, settings.shoulderOffsetY, settings.shoulderOffsetZ };
    Vec3 hmdPos = posFromMatrix(hmdMat);
    Vec3 shoulderWorld = hmdPos + rotateByMatrix(hmdMat, shoulderLocal);

    Vec3 armVec  = ctrlPos - shoulderWorld;
    double armLen = armVec.length();

    if (armLen < settings.minArmLength) { smoother.reset(); return; }

    Vec3 newCtrlPos = shoulderWorld + armVec.normalized() * (armLen * settings.extensionFactor);

    float alpha = clamp_val(1.0f - settings.smoothingFactor, 0.01f, 1.0f);
    Vec3 finalPos = smoother.smooth(newCtrlPos, alpha);

    pose.vecPosition[0] = finalPos.x;
    pose.vecPosition[1] = finalPos.y;
    pose.vecPosition[2] = finalPos.z;

    pose.vecVelocity[0] *= settings.extensionFactor;
    pose.vecVelocity[1] *= settings.extensionFactor;
    pose.vecVelocity[2] *= settings.extensionFactor;
}

// --------------------------------------------------------------------------
// Tracked device wrapper — wraps a real controller and modifies its pose
// --------------------------------------------------------------------------

class CArmExtenderDevice : public vr::ITrackedDeviceServerDriver {
public:
    CArmExtenderDevice(uint32_t realIndex, bool isRight,
                       ArmExtenderSettings* settings)
        : m_realIndex(realIndex), m_isRight(isRight), m_settings(settings) {}

    vr::EVRInitError Activate(uint32_t unObjectId) override {
        m_id = unObjectId;
        return vr::VRInitError_None;
    }
    void Deactivate() override { m_id = vr::k_unTrackedDeviceIndexInvalid; }
    void EnterStandby() override {}
    void* GetComponent(const char*) override { return nullptr; }
    void DebugRequest(const char*, char* buf, uint32_t sz) override { if(sz) buf[0]=0; }

    vr::DriverPose_t GetPose() override { return m_lastPose; }

    void UpdatePose(vr::DriverPose_t pose, const vr::HmdMatrix34_t& hmdMat, bool hmdValid) {
        extendArm(hmdMat, hmdValid, pose, m_isRight, *m_settings, m_smoother);
        m_lastPose = pose;
        if (m_id != vr::k_unTrackedDeviceIndexInvalid)
            vr::VRServerDriverHost()->TrackedDevicePoseUpdated(m_id, m_lastPose, sizeof(m_lastPose));
    }

    uint32_t GetRealIndex() const { return m_realIndex; }

private:
    uint32_t             m_realIndex;
    uint32_t             m_id = vr::k_unTrackedDeviceIndexInvalid;
    bool                 m_isRight;
    ArmExtenderSettings* m_settings;
    vr::DriverPose_t     m_lastPose{};
    PoseSmoother         m_smoother;
};

// --------------------------------------------------------------------------
// Driver provider
// --------------------------------------------------------------------------

using namespace vr;

class CArmExtenderDriver : public IServerTrackedDeviceProvider {
public:
    EVRInitError Init(IVRDriverContext* pDriverContext) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);

        const char* appData = std::getenv("APPDATA");
        if (appData) {
            m_configPath = std::string(appData) + "\\ArmExtender\\arm_extender.cfg";
            m_settings.loadFromFile(m_configPath);
        }

        m_running = true;
        m_watchThread = std::thread([this]() {
            while (m_running) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                if (!m_configPath.empty())
                    m_settings.loadFromFile(m_configPath);
            }
        });

        return VRInitError_None;
    }

    void Cleanup() override {
        m_running = false;
        if (m_watchThread.joinable()) m_watchThread.join();
        VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }

    const char* const* GetInterfaceVersions() override { return k_InterfaceVersions; }

    void RunFrame() override {
        // Get HMD pose
        vr::TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount];
        vr::VRServerDriverHost()->GetRawTrackedDevicePoses(0.f, poses, k_unMaxTrackedDeviceCount);

        const auto& hmdTracked = poses[k_unTrackedDeviceIndex_Hmd];
        bool hmdValid = hmdTracked.bPoseIsValid;
        const vr::HmdMatrix34_t& hmdMat = hmdTracked.mDeviceToAbsoluteTracking;

        // For each tracked controller, get its pose and extend it
        for (uint32_t i = 1; i < k_unMaxTrackedDeviceCount; ++i) {
            if (!poses[i].bDeviceIsConnected) continue;

            ETrackedDeviceClass cls = vr::VRServerDriverHost()->GetTrackedDeviceClass(i);
            if (cls != TrackedDeviceClass_Controller) continue;

            // Determine hand
            char roleBuf[64] = {};
            vr::ETrackedPropertyError err;
            int32_t role = (int32_t)vr::VRProperties()->GetInt32Property(
                vr::VRProperties()->TrackedDeviceToPropertyContainer(i),
                vr::Prop_ControllerRoleHint_Int32, &err);

            bool isRight = (role == (int32_t)TrackedControllerRole_RightHand);
            bool isLeft  = (role == (int32_t)TrackedControllerRole_LeftHand);
            if (!isLeft && !isRight) continue;
            if (isLeft  && !m_settings.enableLeft)  continue;
            if (isRight && !m_settings.enableRight) continue;

            // Build a DriverPose_t from the TrackedDevicePose_t
            vr::DriverPose_t dpose{};
            dpose.poseIsValid     = poses[i].bPoseIsValid;
            dpose.deviceIsConnected = poses[i].bDeviceIsConnected;
            dpose.result          = poses[i].eTrackingResult;
            dpose.vecPosition[0]  = poses[i].mDeviceToAbsoluteTracking.m[0][3];
            dpose.vecPosition[1]  = poses[i].mDeviceToAbsoluteTracking.m[1][3];
            dpose.vecPosition[2]  = poses[i].mDeviceToAbsoluteTracking.m[2][3];
            dpose.vecVelocity[0]  = poses[i].vVelocity.v[0];
            dpose.vecVelocity[1]  = poses[i].vVelocity.v[1];
            dpose.vecVelocity[2]  = poses[i].vVelocity.v[2];
            // Copy rotation from matrix to quaternion
            const auto& m = poses[i].mDeviceToAbsoluteTracking;
            double trace = m.m[0][0] + m.m[1][1] + m.m[2][2];
            double w,x,y,z2;
            if (trace > 0) {
                double s = 0.5/std::sqrt(trace+1.0);
                w = 0.25/s;
                x = (m.m[2][1]-m.m[1][2])*s;
                y = (m.m[0][2]-m.m[2][0])*s;
                z2= (m.m[1][0]-m.m[0][1])*s;
            } else {
                w=0; x=0; y=0; z2=1;
            }
            dpose.qRotation.w=w; dpose.qRotation.x=x;
            dpose.qRotation.y=y; dpose.qRotation.z=z2;
            dpose.qWorldFromDriverRotation.w=1;
            dpose.qDriverFromHeadRotation.w=1;

            auto& smoother = isRight ? m_smootherRight : m_smootherLeft;
            extendArm(hmdMat, hmdValid, dpose, isRight, m_settings, smoother);

            vr::VRServerDriverHost()->TrackedDevicePoseUpdated(i, dpose, sizeof(dpose));
        }
    }

    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

private:
    ArmExtenderSettings m_settings;
    PoseSmoother        m_smootherLeft, m_smootherRight;
    std::thread         m_watchThread;
    std::atomic<bool>   m_running{false};
    std::string         m_configPath;
};

static CArmExtenderDriver g_driver;

extern "C" __declspec(dllexport)
void* HmdDriverFactory(const char* pInterfaceName, int* pReturnCode) {
    if (std::string(pInterfaceName) == IServerTrackedDeviceProvider_Version)
        return &g_driver;
    if (pReturnCode) *pReturnCode = VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
