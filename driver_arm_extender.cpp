/*
 * ArmExtender SteamVR Driver
 * Biomechanically correct arm extension via shoulder-pivot scaling.
 * No windows.h dependency - uses only OpenVR and C++ stdlib.
 */

#include <openvr_driver.h>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
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
// Core arm extension
// --------------------------------------------------------------------------

static void extendArm(
    const vr::TrackedDevicePose_t& hmdPose,
    vr::TrackedDevicePose_t&       controllerPose,
    bool                            isRightHand,
    const ArmExtenderSettings&      settings,
    PoseSmoother&                   smoother)
{
    if (!controllerPose.bPoseIsValid || !hmdPose.bPoseIsValid) {
        smoother.reset();
        return;
    }

    const vr::HmdMatrix34_t& hmdMat  = hmdPose.mDeviceToAbsoluteTracking;
    vr::HmdMatrix34_t&       ctrlMat = controllerPose.mDeviceToAbsoluteTracking;

    float sx = isRightHand ? settings.shoulderOffsetX : -settings.shoulderOffsetX;
    Vec3 shoulderLocal{ sx, settings.shoulderOffsetY, settings.shoulderOffsetZ };
    Vec3 shoulderWorld = posFromMatrix(hmdMat) + rotateByMatrix(hmdMat, shoulderLocal);
    Vec3 ctrlPos = posFromMatrix(ctrlMat);
    Vec3 armVec  = ctrlPos - shoulderWorld;
    double armLen = armVec.length();

    if (armLen < settings.minArmLength) { smoother.reset(); return; }

    Vec3 newCtrlPos = shoulderWorld + armVec.normalized() * (armLen * settings.extensionFactor);

    float alpha = std::clamp(1.0f - settings.smoothingFactor, 0.01f, 1.0f);
    Vec3 finalPos = smoother.smooth(newCtrlPos, alpha);

    ctrlMat.m[0][3] = static_cast<float>(finalPos.x);
    ctrlMat.m[1][3] = static_cast<float>(finalPos.y);
    ctrlMat.m[2][3] = static_cast<float>(finalPos.z);

    controllerPose.vVelocity.v[0] *= settings.extensionFactor;
    controllerPose.vVelocity.v[1] *= settings.extensionFactor;
    controllerPose.vVelocity.v[2] *= settings.extensionFactor;
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------

using namespace vr;

class CArmExtenderDriver : public IServerTrackedDeviceProvider {
public:
    EVRInitError Init(IVRDriverContext* pDriverContext) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);

        // Load config from %APPDATA%\ArmExtender\arm_extender.cfg
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
        vr::TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount];
        vr::VRServerDriverHost()->GetRawTrackedDevicePoses(0.f, poses, k_unMaxTrackedDeviceCount);

        const auto& hmdPose = poses[k_unTrackedDeviceIndex_Hmd];

        for (uint32_t i = 1; i < k_unMaxTrackedDeviceCount; ++i) {
            if (!poses[i].bDeviceIsConnected) continue;
            ETrackedDeviceClass cls = vr::VRSystem()->GetTrackedDeviceClass(i);
            if (cls != TrackedDeviceClass_Controller) continue;

            ETrackedControllerRole role = vr::VRSystem()->GetControllerRoleForTrackedDeviceIndex(i);
            bool isRight = (role == TrackedControllerRole_RightHand);
            bool isLeft  = (role == TrackedControllerRole_LeftHand);
            if (!isLeft && !isRight) continue;
            if (isLeft  && !m_settings.enableLeft)  continue;
            if (isRight && !m_settings.enableRight) continue;

            auto& smoother = isRight ? m_smootherRight : m_smootherLeft;
            extendArm(hmdPose, poses[i], isRight, m_settings, smoother);
            vr::VRServerDriverHost()->TrackedDevicePoseUpdated(i, poses[i], sizeof(vr::TrackedDevicePose_t));
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
