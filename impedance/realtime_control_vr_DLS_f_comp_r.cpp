#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "dsr_realtime_control/realtime_control.hpp"

#include <pthread.h>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <cerrno>
#include <atomic>
#include <mutex>
#include <cmath>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
// [2026-04-12] Pinocchio 분석적 Jacobian 및 J_dot 계산을 위해 추가
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

// ============================================================================
// Rotation Vector Conversion Functions
// ============================================================================
Eigen::Quaterniond rotationVectorToQuaternion(const Eigen::Vector3d& r)
{
    double angle = r.norm();              // 회전각 θ
    if (angle < 1e-9) {
        // 거의 0이면 회전 없음
        return Eigen::Quaterniond::Identity();
    }
    Eigen::Vector3d axis = r / angle;     // 정규화된 회전축
    Eigen::AngleAxisd aa(angle, axis);
    return Eigen::Quaterniond(aa);        // AngleAxis -> Quaternion
}

// rotation vector (axis * angle, rad 단위) → roll, pitch, yaw (rad)
Eigen::Vector3d rotvecToRPY(const Eigen::Vector3d& rotvec)
{
    double angle = rotvec.norm();      // 회전각 (rad)
    Eigen::Vector3d axis(0,0,0);

    if (angle < 1e-12) {
        // 거의 회전 없음 → RPY = 0
        return Eigen::Vector3d::Zero();
    }

    axis = rotvec / angle;             // 회전축 단위벡터

    // Eigen angle-axis -> rotation matrix
    Eigen::AngleAxisd aa(angle, axis);
    Eigen::Matrix3d R = aa.toRotationMatrix();

    // rotation matrix → roll-pitch-yaw (XYZ)
    Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);  // 0:X(roll), 1:Y(pitch), 2:Z(yaw)

    return rpy;
}

// quaternion -> rotation vector (axis-angle)
// q는 단위 쿼터니언이라고 가정
Eigen::Vector3d quaternionToRotationVector(Eigen::Quaterniond q)
{
    // hemisphere 정리: 항상 w >= 0 쪽 사용 (shortest path)
    if (q.w() < 0.0) {
        q.coeffs() *= -1.0;
    }
    q.normalize();

    Eigen::AngleAxisd aa(q);
    double angle = aa.angle();
    Eigen::Vector3d axis = aa.axis();

    if (angle < 1e-9) {
        return Eigen::Vector3d::Zero();
    }

    // rotation vector = θ * u
    return angle * axis;
}


Eigen::Vector3d quaternionToZYX(const Eigen::Quaterniond& q) 
{
    // Quaternion → Rotation Matrix
    Eigen::Matrix3d R = q.toRotationMatrix();
    
    // Rotation Matrix → ZYX Euler angles (Yaw-Pitch-Roll)
    // eulerAngles(2, 1, 0) = Z축, Y축, X축 순서
    Eigen::Vector3d zyx = R.eulerAngles(2, 1, 0);  // [yaw, pitch, roll] (라디안)
    
    return zyx;  // [Z rotation, Y rotation, X rotation]
}

RT_STATE g_stRTState;
std::mutex mtx;
std::atomic_bool first_get(false);

// VR 명령 데이터 (Joint Space)
static std::atomic_bool vr_command_received(false);
static std::mutex vr_mtx;
static Eigen::Vector<double, 6> q_d_vr = Eigen::Vector<double, 6>::Zero();  // VR에서 받은 목표 joint 위치

// Task space command
static std::atomic_bool task_command_received(false);
static Eigen::Vector<double, 6> x_d_vr = Eigen::Vector<double, 6>::Zero();  // Task space command (position only)
static Eigen::Quaterniond quat_d_vr(1.0, 0.0, 0.0, 0.0);  // Task space orientation (quaternion)

// 데이터 로깅용 파일 스트림
std::ofstream data_log_file;
std::atomic_bool logging_enabled(false);
static int log_counter = 0;
static const int LOG_INTERVAL = 10; // 10ms마다 로깅 (1ms * 10)

// Pinocchio model for CRBA computation
std::shared_ptr<pinocchio::Model> g_pinocchio_model;
std::shared_ptr<pinocchio::Data> g_pinocchio_data;
bool g_use_pinocchio_mass = false;

using namespace DRAFramework;
CDRFLEx Drfl;

// ============================================================================
// Forward Kinematics for Doosan M0609 Robot
// ============================================================================
// Modified DH 파라미터 사용
// DH matrix (each row = [a, alpha, d, theta]) -- units: mm & rad
// Link 1: [0,      0,      135,    th1]
// Link 2: [0,      -π/2,   6.25,   th2 - π/2]
// Link 3: [411,    0,      0,      th3 + π/2]
// Link 4: [0,      π/2,    368,    th4]
// Link 5: [0,      -π/2,   0,      th5]
// Link 6: [0,      π/2,    121,    th6]

Eigen::Matrix4d computeTransformationMatrix_ModifiedDH(double a, double alpha, double d, double theta) {
    // Modified DH 파라미터를 사용한 변환 행렬 계산
    // T_mod = | cos(θ)                -sin(θ)               0                a              |
    //         | cos(α)sin(θ)          cos(α)cos(θ)          -sin(α)          -d*sin(α)      |
    //         | sin(α)sin(θ)          sin(α)cos(θ)          cos(α)           d*cos(α)       |
    //         | 0                     0                     0                1              |
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    
    double ct = cos(theta);
    double st = sin(theta);
    double ca = cos(alpha);
    double sa = sin(alpha);
    
    T(0,0) = ct;
    T(0,1) = -st;
    T(0,2) = 0;
    T(0,3) = a;
    
    T(1,0) = ca * st;
    T(1,1) = ca * ct;
    T(1,2) = -sa;
    T(1,3) = -d * sa;
    
    T(2,0) = sa * st;
    T(2,1) = sa * ct;
    T(2,2) = ca;
    T(2,3) = d * ca;
    
    return T;
}

Eigen::Vector<double, 6> forwardKinematics_M0609(const Eigen::Vector<double, 6>& q) {
    // M0609 로봇의 Modified DH 파라미터
    // DH matrix (each row = [a, alpha, d, theta]) -- units: mm & rad
    // Link 1: [0,      0,      135,    th1]
    // Link 2: [0,      -π/2,   6.25,   th2 - π/2]
    // Link 3: [411,    0,      0,      th3 + π/2]
    // Link 4: [0,      π/2,    368,    th4]
    // Link 5: [0,      -π/2,   0,      th5]
    // Link 6: [0,      π/2,    121,    th6]
    
    // Modified DH 파라미터 (단위: mm)
    double a1 = 0.0,      alpha1 = 0.0,      d1 = 135.0,    theta1 = q[0];
    double a2 = 0.0,      alpha2 = -M_PI/2,  d2 = 6.25,     theta2 = q[1] - M_PI/2;
    double a3 = 411.0,    alpha3 = 0.0,      d3 = 0.0,      theta3 = q[2] + M_PI/2;
    double a4 = 0.0,      alpha4 = M_PI/2,   d4 = 368.0,    theta4 = q[3];
    double a5 = 0.0,      alpha5 = -M_PI/2,  d5 = 0.0,      theta5 = q[4];
    double a6 = 0.0,      alpha6 = M_PI/2,   d6 = 121.0,    theta6 = q[5];
    
    // 각 링크의 변환 행렬 계산
    Eigen::Matrix4d T1 = computeTransformationMatrix_ModifiedDH(a1, alpha1, d1, theta1);
    Eigen::Matrix4d T2 = computeTransformationMatrix_ModifiedDH(a2, alpha2, d2, theta2);
    Eigen::Matrix4d T3 = computeTransformationMatrix_ModifiedDH(a3, alpha3, d3, theta3);
    Eigen::Matrix4d T4 = computeTransformationMatrix_ModifiedDH(a4, alpha4, d4, theta4);
    Eigen::Matrix4d T5 = computeTransformationMatrix_ModifiedDH(a5, alpha5, d5, theta5);
    Eigen::Matrix4d T6 = computeTransformationMatrix_ModifiedDH(a6, alpha6, d6, theta6);
    
    // 전체 변환 행렬: TCP = T1 * T2 * T3 * T4 * T5 * T6
    Eigen::Matrix4d T_total = T1 * T2 * T3 * T4 * T5 * T6;
    
    // TCP 위치 추출 (이미 mm 단위)
    Eigen::Vector3d position;
    position(0) = T_total(0,3);  // x (mm)
    position(1) = T_total(1,3);  // y (mm)
    position(2) = T_total(2,3);  // z (mm)
    
    // TCP 자세 추출 (Rotation Vector - Axis-Angle representation)
    // Rotation matrix to rotation vector using Eigen
    Eigen::Matrix3d R = T_total.block<3,3>(0,0);
    
    // Convert rotation matrix to axis-angle
    Eigen::AngleAxisd angle_axis(R);
    Eigen::Vector3d rotation_vector = angle_axis.axis() * angle_axis.angle();
    
    // 결과를 degrees로 변환 (Doosan 로봇은 degree 단위 사용)
    double rx = rotation_vector(0) * 180.0 / M_PI;
    double ry = rotation_vector(1) * 180.0 / M_PI;
    double rz = rotation_vector(2) * 180.0 / M_PI;
    
    // TCP pose: [x(mm), y(mm), z(mm), rx(deg), ry(deg), rz(deg)]
    Eigen::Vector<double, 6> tcp_pose;
    tcp_pose << position(0), position(1), position(2), rx, ry, rz;
    
    return tcp_pose;
}
// ============================================================================

// VR Joint Command Subscriber Node
class VRJointCommandNode : public rclcpp::Node
{
public:
    VRJointCommandNode() : Node("VRJointCommandSubscriberR")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            // "/joint_state_command",  // Topic name
            "right_dsr_joint_controller/joint_state_command",  // Topic name
            10,
            std::bind(&VRJointCommandNode::joint_command_callback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "VR Joint Command Subscriber initialized on right_dsr_joint_controller/joint_state_command");
    }

private:
    void joint_command_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (msg->position.size() < 6) {
            RCLCPP_WARN(this->get_logger(), "Received joint command with less than 6 joints");
            return;
        }

        std::lock_guard<std::mutex> lock(vr_mtx);
        for (size_t i = 0; i < 6; i++) {
            q_d_vr(i) = msg->position[i];  // radians
        }
        vr_command_received = true;

        // RCLCPP_INFO(this->get_logger(), "VR Command: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
        //     q_d_vr(0), q_d_vr(1), q_d_vr(2), q_d_vr(3), q_d_vr(4), q_d_vr(5));
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};

// Task Space Command Subscriber Node
class TaskSpaceCommandNode : public rclcpp::Node
{
public:
    TaskSpaceCommandNode() : Node("TaskSpaceCommandSubscriberR")
    {
        subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/right_dsr_controller/task_space_command",
            10,
            std::bind(&TaskSpaceCommandNode::task_command_callback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "Task Space Command Subscriber initialized");
    }

private:
    void task_command_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(vr_mtx);
        
        // Position (mm)
        x_d_vr(0) = msg->pose.position.x;
        x_d_vr(1) = msg->pose.position.y;
        x_d_vr(2) = msg->pose.position.z;
        
        // Orientation (store quaternion directly)
        quat_d_vr = Eigen::Quaterniond(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z
        );
        quat_d_vr.normalize();  // Ensure unit quaternion
        
        task_command_received = true;
        
        // RCLCPP_INFO(this->get_logger(), "Task Command: pos=[%.1f, %.1f, %.1f], ori=[%.1f, %.1f, %.1f]",
        //     x_d_vr(0), x_d_vr(1), x_d_vr(2), x_d_vr(3), x_d_vr(4), x_d_vr(5));
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
};

// CSV Calculated Pose Command Subscriber Node
class CsvPoseCommandNode : public rclcpp::Node
{
public:
    CsvPoseCommandNode() : Node("CsvPoseCommandSubscriberR")
    {
        subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/calculated_pose",
            10,
            std::bind(&CsvPoseCommandNode::pose_command_callback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "CSV Pose Command Subscriber initialized on /calculated_pose");
    }

private:
    void pose_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) // VR Tracker pose 받아옴, from, servo_left_imp.py
    {
        if (msg->data.size() < 6) {
            RCLCPP_WARN(this->get_logger(), "Received pose command with less than 6 elements");
            return;
        }

        std::lock_guard<std::mutex> lock(vr_mtx);
        
        // Position (mm)
        x_d_vr(0) = msg->data[0];
        x_d_vr(1) = msg->data[1];
        x_d_vr(2) = msg->data[2];
        
        // Orientation (ZYX Euler angles in degrees -> Quaternion)
        double a_des = msg->data[3] * M_PI / 180.0; // Z rotation
        double b_des = msg->data[4] * M_PI / 180.0; // Y rotation
        double c_des = msg->data[5] * M_PI / 180.0; // X rotation
        
        Eigen::Matrix3d R_des =
            (Eigen::AngleAxisd(a_des, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(b_des, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(c_des, Eigen::Vector3d::UnitX())).toRotationMatrix();
             
        quat_d_vr = Eigen::Quaterniond(R_des);
        quat_d_vr.normalize();  // Ensure unit quaternion
        
        task_command_received = true;
    }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
};

ReadDataRtNode::ReadDataRtNode() : Node("ReadDataRtR")
{
    client_ = this->create_client<dsr_msgs2::srv::ReadDataRt>("/dsr01/realtime/read_data_rt");
    client_thread_ = std::thread(std::bind(&ReadDataRtNode::ReadDataRtClient, this));

    // auto timer_callback = [this]() -> void 
    // {
    //     auto context_switches = context_switches_counter.get();
    //     if (context_switches > 0L) 
    //     {
    //       RCLCPP_WARN(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     } 
    //     else 
    //     {
    //       RCLCPP_INFO(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     }
    // };
    // context_timer_ = this->create_wall_timer(std::chrono::milliseconds(500), timer_callback);
}

TorqueRtNode::TorqueRtNode() : Node("TorqueRtR")
{
    publisher_  = this->create_publisher<dsr_msgs2::msg::TorqueRtStream>("/dsr01/torque_rt_stream",10);
    // Publisher for impedance force (6-element) as Float64MultiArray
    imp_force_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/imp_force", 10);
    // Publishers for delta_x, delta_x_dot and F_e
    delta_x_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/delta_x", 10);
    delta_x_dot_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/delta_x_dot", 10);
    F_e_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/F_e", 10);
    F_e_raw_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/F_e_raw", 10);
    current_pose_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/current_pose", 10);
    desired_pose_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/desired_pose", 10);
    // Publisher for J_dot (6x6 flattened)
    J_dot_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/J_dot", 10);
    // Publisher for J_dot * q_dot (6-element)
    Jdot_qdot_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/Jdot_qdot", 10);
    // Publishers for B_d*delta_x_dot and K_d*delta_x
    B_delta_x_dot_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/B_delta_x_dot", 10);
    K_delta_x_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/K_delta_x", 10);
    // Publishers for intermediate terms (term1..term4)
    term1_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/term1", 10);
    term2_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/term2", 10);
    term3_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/term3", 10);
    term4_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/term4", 10);
    term6_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/term6", 10);
    term7_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/term7", 10);
    // Publishers for acceleration_ref and J_inv * acceleration_ref
    acceleration_ref_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/acceleration_ref", 10);
    Jinv_acc_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/Jinv_acc", 10);
    // Publisher for rpy_test
    // rpy_test_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/rpy_test", 10);
    timer_      = this->create_wall_timer(std::chrono::microseconds(1000),std::bind(&TorqueRtNode::TorqueRtStreamPublisher,this));

    // Create subscriber for aggregated hand wrench
    aggregated_wrench_sub_ = this->create_subscription<sensor_msgs::msg::MultiDOFJointState>(
        "/right_aggregated_wrench", 10,
        std::bind(&TorqueRtNode::aggregatedWrenchCallback, this, std::placeholders::_1));

    // 파라미터 선언 및 기본값 설정
    this->declare_parameter("impedance.mass.linear", std::vector<double>{20.0, 20.0, 20.0});
    this->declare_parameter("impedance.mass.angular", std::vector<double>{20.0, 20.0, 20.0});
    this->declare_parameter("impedance.damping.linear", std::vector<double>{0.01, 0.01, 0.01});
    this->declare_parameter("impedance.damping.angular", std::vector<double>{0.01, 0.01, 0.01});
    this->declare_parameter("impedance.stiffness.linear", std::vector<double>{10.0, 10.0, 10.0});
    this->declare_parameter("impedance.stiffness.angular", std::vector<double>{10.0, 10.0, 10.0});
    
    this->declare_parameter("control.torque_limits", std::vector<double>{25.0, 25.0, 25.0, 25.0, 25.0, 25.0});
    this->declare_parameter("control.print_interval", 50);
    this->declare_parameter("control.log_interval", 10);
    this->declare_parameter("control.jacobian_lpf_alpha", 0.1);
    this->declare_parameter("control.dls_lambda", 0.01);
    this->declare_parameter("control.force_lpf_alpha", 0.1);
    this->declare_parameter("control.mass_matrix_scale", std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    
    this->declare_parameter("tool.mass", 1.5);
    this->declare_parameter("tool.offset_z", 0.2);
    this->declare_parameter("tool.gravity_direction", std::vector<double>{0.0, -0.707, 0.707});
    
    this->declare_parameter("desired_position.position", std::vector<double>{-450.0, 50.0, 450.0});
    this->declare_parameter("desired_position.orientation", std::vector<double>{0.0, -60.0, -135.0});
    
    this->declare_parameter("pinocchio.use_crba_mass_matrix", false);
    this->declare_parameter("pinocchio.urdf_path", "");

    // yaml에서 파라미터 읽기
    this->get_parameter("impedance.mass.linear", m_lin);
    this->get_parameter("impedance.mass.angular", m_ang);
    this->get_parameter("impedance.damping.linear", c_lin);
    this->get_parameter("impedance.damping.angular", c_ang);
    this->get_parameter("impedance.stiffness.linear", k_lin);
    this->get_parameter("impedance.stiffness.angular", k_ang);
    this->get_parameter("control.torque_limits", torque_limit);
    // get force LPF alpha
    this->get_parameter("control.force_lpf_alpha", force_lpf_alpha_);
    // get DLS damping factor lambda
    this->get_parameter("control.dls_lambda", dls_lambda_);
    // get mass matrix diagonal scaling factors
    this->get_parameter("control.mass_matrix_scale", mass_matrix_scale);

    // Create data directory with absolute path
    std::string package_path = "/home/vision/doosan_ws/src/doosan-robot2/dsr_example2/dsr_realtime_control";
    std::string data_dir = package_path + "/data";
    
    if (mkdir(data_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create directory: %s", data_dir.c_str());
    }

    // 데이터 로깅 파일 초기화
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::stringstream filename;
    filename << data_dir << "/impedance_control_data_vr_r_" 
             << std::put_time(&tm, "%Y%m%d_%H%M%S") 
             << ".csv";
    
    data_log_file.open(filename.str());
    if (data_log_file.is_open()) {
        // CSV 헤더 작성
        data_log_file << "timestamp,";
        data_log_file << "G_q_0,G_q_1,G_q_2,G_q_3,G_q_4,G_q_5,";
        data_log_file << "tau_calc_0,tau_calc_1,tau_calc_2,tau_calc_3,tau_calc_4,tau_calc_5,";
        data_log_file << "tau_lim_0,tau_lim_1,tau_lim_2,tau_lim_3,tau_lim_4,tau_lim_5,";
        data_log_file << "ext_jt_0,ext_jt_1,ext_jt_2,ext_jt_3,ext_jt_4,ext_jt_5,";
        data_log_file << "impedance_force_0,impedance_force_1,impedance_force_2,impedance_force_3,impedance_force_4,impedance_force_5,";
        data_log_file << "acceleration_ref_0,acceleration_ref_1,acceleration_ref_2,acceleration_ref_3,acceleration_ref_4,acceleration_ref_5,";
        data_log_file << "M_e_J_inv_acc_0,M_e_J_inv_acc_1,M_e_J_inv_acc_2,M_e_J_inv_acc_3,M_e_J_inv_acc_4,M_e_J_inv_acc_5,";
        data_log_file << "C_q_dot_0,C_q_dot_1,C_q_dot_2,C_q_dot_3,C_q_dot_4,C_q_dot_5,";
        data_log_file << "G_q_term_0,G_q_term_1,G_q_term_2,G_q_term_3,G_q_term_4,G_q_term_5,";
        data_log_file << "J_T_F_e_0,J_T_F_e_1,J_T_F_e_2,J_T_F_e_3,J_T_F_e_4,J_T_F_e_5,";
        data_log_file << "desired_pose_0,desired_pose_1,desired_pose_2,desired_pose_3,desired_pose_4,desired_pose_5,";
        data_log_file << "current_pose_0,current_pose_1,current_pose_2,current_pose_3,current_pose_4,current_pose_5,";
        data_log_file << "F_e_raw_0,F_e_raw_1,F_e_raw_2,F_e_raw_3,F_e_raw_4,F_e_raw_5,";
        data_log_file << "F_e_lpf_0,F_e_lpf_1,F_e_lpf_2,F_e_lpf_3,F_e_lpf_4,F_e_lpf_5,";
        data_log_file << "J_dot_q_dot_0,J_dot_q_dot_1,J_dot_q_dot_2,J_dot_q_dot_3,J_dot_q_dot_4,J_dot_q_dot_5,";
        data_log_file << "B_delta_x_dot_0,B_delta_x_dot_1,B_delta_x_dot_2,B_delta_x_dot_3,B_delta_x_dot_4,B_delta_x_dot_5,";
        data_log_file << "K_delta_x_0,K_delta_x_1,K_delta_x_2,K_delta_x_3,K_delta_x_4,K_delta_x_5,";
        data_log_file << "x_dot_current_0,x_dot_current_1,x_dot_current_2,x_dot_current_3,x_dot_current_4,x_dot_current_5,";
        data_log_file << "q_dot_current_0,q_dot_current_1,q_dot_current_2,q_dot_current_3,q_dot_current_4,q_dot_current_5";
        data_log_file << std::endl;
        logging_enabled = true;
        RCLCPP_INFO(this->get_logger(), "Data logging started: %s", filename.str().c_str());
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open log file: %s", filename.str().c_str());
    }

    // Initialize Pinocchio model for CRBA mass matrix computation
    std::string urdf_path;
    this->get_parameter("pinocchio.use_crba_mass_matrix", g_use_pinocchio_mass);
    this->get_parameter("pinocchio.urdf_path", urdf_path);
    
    if (g_use_pinocchio_mass) {
        if (urdf_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Pinocchio URDF path is empty! Disabling CRBA mass matrix computation.");
            g_use_pinocchio_mass = false;
        } else {
            try {
                g_pinocchio_model = std::make_shared<pinocchio::Model>();
                pinocchio::urdf::buildModel(urdf_path, *g_pinocchio_model);
                g_pinocchio_data = std::make_shared<pinocchio::Data>(*g_pinocchio_model);
                RCLCPP_INFO(this->get_logger(), "Pinocchio model loaded successfully from: %s", urdf_path.c_str());
                RCLCPP_INFO(this->get_logger(), "Model has %d DOFs", g_pinocchio_model->nq);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load Pinocchio model: %s", e.what());
                g_use_pinocchio_mass = false;
            }
        }
    }

    // auto timer_callback = [this]() -> void 
    // {
    //     auto context_switches = context_switches_counter.get();
    //     if (context_switches > 0L) 
    //     {
    //       RCLCPP_WARN(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     } 
    //     else 
    //     {
    //       RCLCPP_INFO(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     }
    // };
    // context_timer_ = this->create_wall_timer(std::chrono::milliseconds(500), timer_callback);
}

ServojRtNode::ServojRtNode() : Node("ServojRtR")
{
    publisher_  = this->create_publisher<dsr_msgs2::msg::ServojRtStream>("/dsr01/servoj_rt_stream",10);
    timer_      = this->create_wall_timer(std::chrono::microseconds(1000),std::bind(&ServojRtNode::ServojRtStreamPublisher,this));

    // auto timer_callback = [this]() -> void 
    // {
    //     auto context_switches = context_switches_counter.get();
    //     if (context_switches > 0L) 
    //     {
    //       RCLCPP_WARN(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     } 
    //     else 
    //     {
    //       RCLCPP_INFO(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     }
    // };
    // context_timer_ = this->create_wall_timer(std::chrono::milliseconds(500), timer_callback);
}

ServolRtNode::ServolRtNode() : Node("ServolRtR")
{
    publisher_  = this->create_publisher<dsr_msgs2::msg::ServolRtStream>("/dsr01/servol_rt_stream",10);
    timer_      = this->create_wall_timer(std::chrono::microseconds(1000),std::bind(&ServolRtNode::ServolRtStreamPublisher,this));

    // auto timer_callback = [this]() -> void 
    // {
    //     auto context_switches = context_switches_counter.get();
    //     if (context_switches > 0L) 
    //     {
    //       RCLCPP_WARN(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     } 
    //     else 
    //     {
    //       RCLCPP_INFO(this->get_logger(), "Involuntary context switches: '%lu'", context_switches);
    //     }
    // };
    // context_timer_ = this->create_wall_timer(std::chrono::milliseconds(500), timer_callback);
}

ReadDataRtNode::~ReadDataRtNode()
{
    if(client_thread_.joinable())
    {
        client_thread_.join();
        RCLCPP_INFO(this->get_logger(), "client_thread_.joined");
    }
    RCLCPP_INFO(this->get_logger(), "ReadDataRt client shut down");
}
TorqueRtNode::~TorqueRtNode()
{
    // 데이터 로깅 파일 닫기
    if (data_log_file.is_open()) {
        data_log_file.close();
        logging_enabled = false;
        RCLCPP_INFO(this->get_logger(), "Data logging file closed");
    }
    RCLCPP_INFO(this->get_logger(), "TorqueRt publisher shut down");
}

void TorqueRtNode::aggregatedWrenchCallback(const sensor_msgs::msg::MultiDOFJointState::SharedPtr msg)
{
    if (msg->wrench.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                             "Received empty wrench message");
        return;
    }

    std::lock_guard<std::mutex> lock(hand_wrench_mutex_);
    
    // Extract force and torque from aggregated wrench
    hand_force_[0] = msg->wrench[0].force.x;
    hand_force_[1] = msg->wrench[0].force.y;
    hand_force_[2] = msg->wrench[0].force.z;
    
    hand_torque_[0] = msg->wrench[0].torque.x;
    hand_torque_[1] = msg->wrench[0].torque.y;
    hand_torque_[2] = msg->wrench[0].torque.z;
    
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                          "Received hand wrench - Force: [%.2f, %.2f, %.2f] N, Torque: [%.2f, %.2f, %.2f] Nm",
                          hand_force_[0], hand_force_[1], hand_force_[2],
                          hand_torque_[0], hand_torque_[1], hand_torque_[2]);
}

ServojRtNode::~ServojRtNode()
{
    RCLCPP_INFO(this->get_logger(), "ServojRt publisher shut down");
}
ServolRtNode::~ServolRtNode()
{
    RCLCPP_INFO(this->get_logger(), "ServolRt publisher shut down");
}

void ReadDataRtNode::ReadDataRtClient()
{
    rclcpp::Rate rate(3000);
    while(rclcpp::ok())
    {
        rate.sleep();
        if (!client_->wait_for_service(std::chrono::seconds(1)))
        {
            RCLCPP_WARN(this->get_logger(), "Waiting for the server to be up...");
            continue;
        }
        auto request = std::make_shared<dsr_msgs2::srv::ReadDataRt::Request>();
        auto future = client_->async_send_request(request);
        // RCLCPP_INFO(this->get_logger(), "ReadDataRt Service Request");
        try
        {
            auto response = future.get();
            if(!first_get)
            {
                first_get=true;
            }
            // RCLCPP_INFO(this->get_logger(), "ReadDataRt Service Response");
            g_stRTState.time_stamp = response->data.time_stamp;
            for(int i=0; i<6; i++)
            {
                g_stRTState.actual_joint_position[i] = response->data.actual_joint_position[i];
                g_stRTState.actual_joint_velocity[i] = response->data.actual_joint_velocity[i];
                g_stRTState.actual_tcp_position[i] = response->data.actual_tcp_position[i];
                g_stRTState.gravity_torque[i] = response->data.gravity_torque[i];
                g_stRTState.external_joint_torque[i] = response->data.external_joint_torque[i];
                g_stRTState.external_tcp_force[i] = response->data.external_tcp_force[i];
                g_stRTState.actual_joint_torque[i] = response->data.actual_joint_torque[i];

            }
            for(int i = 0; i < 6; i++)
            {
                for(int j = 0; j < 6; j++)
                {
                    g_stRTState.coriolis_matrix[i][j] = response->data.coriolis_matrix[i].data[j];
                    // g_stRTState.mass_matrix[i][j] = response->data.mass_matrix[i].data[j];
                    g_stRTState.jacobian_matrix[i][j] = response->data.jacobian_matrix[i].data[j];
                }
            }
            
            // Compute mass matrix using Pinocchio CRBA if enabled
            if (g_use_pinocchio_mass && g_pinocchio_model && g_pinocchio_data) {
                // Convert joint positions from degrees to radians for Pinocchio
                Eigen::VectorXd q_pin(g_pinocchio_model->nq);
                for(int i = 0; i < std::min(6, (int)g_pinocchio_model->nq); i++) {
                    q_pin(i) = g_stRTState.actual_joint_position[i] * M_PI / 180.0;
                }
                
                // Compute mass matrix using CRBA algorithm
                pinocchio::crba(*g_pinocchio_model, *g_pinocchio_data, q_pin);
                g_pinocchio_data->M.triangularView<Eigen::StrictlyLower>() = 
                    g_pinocchio_data->M.triangularView<Eigen::StrictlyUpper>().transpose();
                
                // Update global state with Pinocchio mass matrix
                for(int i = 0; i < 6; i++) {
                    for(int j = 0; j < 6; j++) {
                        g_stRTState.mass_matrix[i][j] = g_pinocchio_data->M(i, j);
                    }
                }
                
                static int debug_counter = 0;
                if(debug_counter % 1000 == 0) {
                    RCLCPP_INFO(this->get_logger(), "CRBA computed - M(0,0) = %f", g_pinocchio_data->M(0, 0));
                }
                debug_counter++;
            } else {
                static int warn_counter = 0;
                if(warn_counter % 1000 == 0) {
                    RCLCPP_WARN(this->get_logger(), "Pinocchio CRBA not used: use_pinocchio=%d, model=%p, data=%p", 
                        g_use_pinocchio_mass, (void*)g_pinocchio_model.get(), (void*)g_pinocchio_data.get());
                }
                warn_counter++;
            }
            
            // RCLCPP_INFO(this->get_logger(), "time stamp : %f",g_stRTState.time_stamp);
        }
        catch(const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Service call failed");
        }
    }
}

void TorqueRtNode::TorqueRtStreamPublisher()
{
    // </----- your control logic start ----->
    
    // === Current State Variables ===
    Eigen::Vector<double, 6> q_current, q_dot_current;           // 현재 관절 위치, 속도
    Eigen::Vector<double, 6> x_current, x_dot_current;           // 현재 TCP 위치, 속도
    
    // === Desired State Variables ===
    Eigen::Vector<double, 6> x_d, x_dot_d, x_ddot_d;            // 목표 TCP 위치, 속도, 가속도
    static Eigen::Vector<double, 6> x_d_initial;                // 초기 목표 위치 (한 번만 설정)
    static Eigen::Quaterniond quat_d_initial(1.0, 0.0, 0.0, 0.0); // 초기 목표 자세 (quaternion)
    static bool x_d_initialized = false;                        // 초기화 플래그
    
    // === Error Variables ===
    Eigen::Vector<double, 6> delta_x, delta_x_dot;              // 위치 오차, 속도 오차
    
    // === Matrices ===
    Eigen::Matrix<double, 6, 6> J, J_T, J_inv;                  // 자코비안, 전치, 역행렬
    Eigen::Matrix<double, 6, 6> J_dot;                          // 자코비안 미분
    Eigen::Matrix<double, 6, 6> M_e;                            // End-effector 질량행렬
    Eigen::Matrix<double, 6, 6> M_q;                            // Joint space 질량행렬
    Eigen::Matrix<double, 6, 6> C_q;                            // 코리올리스 행렬
    
    // === Desired Impedance Parameters ===
    Eigen::Matrix<double, 6, 6> M_d, B_d, K_d;                  // 목표 질량, 댐핑, 강성
    
    // === Force Variables ===
    Eigen::Vector<double, 6> F_e;                               // 외부 힘
    Eigen::Vector<double, 6> G_q;                               // 중력 토크
    Eigen::Vector<double, 6> tau_impedance;                     // 최종 임피던스 토크
    Eigen::Vector<double, 6> F_tool_gravity;                    // 툴 중력 힘

    // temporary raw external force vector (from sensor)
    Eigen::Vector<double, 6> F_e_raw;
    
    for(int i=0; i<6; i++)
    {
        mtx.lock();

        // 현재 관절 상태
        q_current(i) = g_stRTState.actual_joint_position[i];
        // q_dot_current(i) = g_stRTState.actual_joint_velocity[i];
        q_dot_current(i) = g_stRTState.actual_joint_velocity[i];  // [TEMP] raw 값 먼저 저장 (아래에서 LPF 적용)
        
        // 외부 힘/토크 (원시 데이터) — apply sign convention
        if (i < 3) {
            F_e_raw(i) = g_stRTState.external_tcp_force[i];
        } else {
            F_e_raw(i) = g_stRTState.external_tcp_force[i];
        }
        G_q(i) = g_stRTState.gravity_torque[i];
        
        // 행렬들 복사
        for(int j=0; j<6; j++)
        {
            J(i,j) = g_stRTState.jacobian_matrix[i][j];
            M_q(i,j) = g_stRTState.mass_matrix[i][j];
            C_q(i,j) = g_stRTState.coriolis_matrix[i][j];
        }

        trq_g[i]    =   g_stRTState.gravity_torque[i];
        ext_JT[i]   =   g_stRTState.external_joint_torque[i];
        ext_tcp[i]  =   g_stRTState.external_tcp_force[i];
        mtx.unlock();
    }
    
    // ======== [TEMP] q_dot_current Low-Pass Filter ========
    // 임시 LPF: q_dot_current에 1차 저역통과 필터 적용
    // alpha_qdot가 작을수록 더 부드럽게, 클수록 원본에 가까움
    // 제거 시 이 블록 전체를 삭제하면 됨
    // {
    //     static Eigen::Vector<double, 6> q_dot_lpf = Eigen::Vector<double, 6>::Zero();
    //     static bool q_dot_lpf_initialized = false;
    //     const double alpha_qdot = 0.95;  // [TEMP] LPF 계수 (조정 가능: 0.01~1.0)

    //     if (!q_dot_lpf_initialized) {
    //         q_dot_lpf = q_dot_current;
    //         q_dot_lpf_initialized = true;
    //     } else {
    //         q_dot_lpf = alpha_qdot * q_dot_current + (1.0 - alpha_qdot) * q_dot_lpf;
    //     }
    //     q_dot_current = q_dot_lpf;  // [TEMP] 필터링된 값으로 덮어쓰기
    // }
    // ======== [TEMP] q_dot_current Low-Pass Filter END ========
    
    // Add hand wrench to external force/torque
    {
        std::lock_guard<std::mutex> lock(hand_wrench_mutex_);
        // Sum hand force with wrist FT sensor force
        F_e_raw(0) += 3 * hand_force_[0];  // Fx
        F_e_raw(1) += 3 * hand_force_[1];  // Fy
        F_e_raw(2) += 3 * hand_force_[2];  // Fz
        // Sum hand torque with wrist FT sensor torque
        F_e_raw(3) += 3 * hand_torque_[0]; // Tx
        F_e_raw(4) += 3 * hand_torque_[1]; // Ty
        F_e_raw(5) += 3 * hand_torque_[2]; // Tz
    }
    
    // === Mass Matrix Diagonal Scaling ===
    // YAML에서 설정한 계수를 mass matrix의 대각 성분에 곱함
    for(int i = 0; i < 6; i++)
    {
        M_q(i, i) *= mass_matrix_scale[i];
    }

    // === Bias Calibration for F_e ===
    static Eigen::Vector<double, 6> F_e_offset = Eigen::Vector<double, 6>::Zero();
    static bool F_e_offset_initialized = false;

    if (!F_e_offset_initialized && first_get) 
    {
        // Save the initial force as an offset baseline
        F_e_offset = F_e_raw;
        F_e_offset_initialized = true;
        printf("Initial F_e_offset set to: [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
               F_e_offset(0), F_e_offset(1), F_e_offset(2), 
               F_e_offset(3), F_e_offset(4), F_e_offset(5));
    }
    
    // Subtract the offset from the raw reading BEFORE the Low Pass Filter
    if (first_get) {
        F_e_raw = F_e_raw - F_e_offset;
    }

    // === Low-pass filter F_e (external force) ===
    static Eigen::Vector<double,6> F_e_lpf = Eigen::Vector<double,6>::Zero();
    static bool F_e_lpf_initialized = false;
    double alpha_f = force_lpf_alpha_;
    if (!F_e_lpf_initialized && F_e_offset_initialized) {
        F_e_lpf = F_e_raw;
        F_e_lpf_initialized = true;
    } else if (F_e_lpf_initialized) {
        F_e_lpf = alpha_f * F_e_raw + (1.0 - alpha_f) * F_e_lpf;
    }

    // use filtered external force for control and publishing
    F_e = F_e_lpf;
    
    // === Compute Derived Quantities ===
    J_T = J.transpose();
    // DLS (Damped Least Squares) pseudo-inverse: J_DLS = J^T * (J*J^T + λ²I)^{-1}
    {
        double lambda = dls_lambda_;
        Eigen::Matrix<double, 6, 6> JJT = J * J_T;
        Eigen::Matrix<double, 6, 6> lambda2_I = (lambda * lambda) * Eigen::Matrix<double, 6, 6>::Identity();
        J_inv = J_T * (JJT + lambda2_I).inverse();
    }
    
    // End-effector mass matrix: M_e = (J * M_q^-1 * J^T)^-1
    // M_e = (J * M_q.inverse() * J_T).inverse();

    // === Current TCP State (Forward Kinematics 필요) ===
    // C 배열을 Eigen 벡터로 복사
    for(int i = 0; i < 6; i++)
    {
        x_current(i) = g_stRTState.actual_tcp_position[i];
    }
    x_dot_current = J * q_dot_current;                 // TCP 속도
    
    // === FK 검증: q_current로 계산한 TCP와 실제 TCP 비교 ===
    static int fk_verify_counter = 0;
    static const int FK_VERIFY_INTERVAL = 500; // 500ms마다 검증 (1ms * 500)
    fk_verify_counter++;
    if (fk_verify_counter >= FK_VERIFY_INTERVAL) {
        fk_verify_counter = 0;
        
        // 현재 관절 각도를 deg에서 rad로 변환
        Eigen::Vector<double, 6> q_current_rad = q_current * M_PI / 180.0;
        
        // 현재 관절 각도로 FK 계산
        Eigen::Vector<double, 6> x_fk_computed = forwardKinematics_M0609(q_current_rad);
        
        // 실제 TCP와 FK 계산 결과 비교
        Eigen::Vector<double, 6> fk_error = x_fk_computed - x_current;
        
        printf("\n=== FK Verification ===\n");
        printf("q_current (rad):   [%8.4f][%8.4f][%8.4f][%8.4f][%8.4f][%8.4f]\n",
               q_current(0), q_current(1), q_current(2), q_current(3), q_current(4), q_current(5));
        printf("x_current (real):  [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f] (mm, deg)\n",
               x_current(0), x_current(1), x_current(2), x_current(3), x_current(4), x_current(5));
        printf("x_fk (computed):   [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f] (mm, deg)\n",
               x_fk_computed(0), x_fk_computed(1), x_fk_computed(2), 
               x_fk_computed(3), x_fk_computed(4), x_fk_computed(5));
        printf("FK Error:          [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n",
               fk_error(0), fk_error(1), fk_error(2), fk_error(3), fk_error(4), fk_error(5));
        printf("Position Error Norm: %.3f mm\n", 
               sqrt(fk_error(0)*fk_error(0) + fk_error(1)*fk_error(1) + fk_error(2)*fk_error(2)));
        printf("Orientation Error Norm: %.3f deg\n", 
               sqrt(fk_error(3)*fk_error(3) + fk_error(4)*fk_error(4) + fk_error(5)*fk_error(5)));
        printf("======================\n\n");
    }
    
    // === Tool Weight Compensation ===
    // 툴 무게: 1.5kg, TCP에서 z축으로 200mm 앞에 위치
    double tool_mass = 1.5;  // kg
    double tool_offset_z = 0.2;  // 200mm = 0.2m
    double gravity_magnitude = 9.81;   // m/s²
    
    // TCP 자세 (a, b, c) 추출 - ZYZ Euler angles (degrees)
    double a_deg = x_current(3);  // alpha (첫 번째 z축 회전)
    double b_deg = x_current(4);  // beta (y축 회전) 
    double c_deg = x_current(5);  // gamma (두 번째 z축 회전)
    
    // degrees를 radians로 변환
    double a_rad = a_deg * M_PI / 180.0;
    double b_rad = b_deg * M_PI / 180.0;
    double c_rad = c_deg * M_PI / 180.0;
    
    // 회전 행렬 생성 (ZYZ Euler angle 순서)
    // R = Rz(a) * Ry(b) * Rz(c)
    Eigen::Matrix3d R_z, R_y, R_x;
    
    // 첫 번째 Z 회전 (alpha)
    R_z << cos(a_rad), -sin(a_rad), 0,
            sin(a_rad),  cos(a_rad), 0,
            0,           0,          1;
    
    // Y 회전 (beta)
    R_y << cos(b_rad),  0, sin(b_rad),
           0,           1, 0,
           -sin(b_rad), 0, cos(b_rad);
    
    // 두 번째 Z 회전 (gamma)
    // R_z2 << cos(c_rad), -sin(c_rad), 0,
    //         sin(c_rad),  cos(c_rad), 0,
    //         0,           0,          1;
    R_x << 1,          0,           0,
            0, cos(c_rad), -sin(c_rad),
            0, sin(c_rad),  cos(c_rad);
    
    // 최종 회전 행렬: R = Rz(a) * Ry(b) * Rz(c)
    Eigen::Matrix3d R_tcp = R_z * R_y * R_x;
    
    // 베이스 좌표계에서 중력 벡터 (y축 45도, z축 45도 방향)
    // y축 45도: yz 평면에서 45도 회전
    // z축 45도: xz 평면에서 추가적인 회전
    // double gravity_y_angle = -45.0 * M_PI / 180.0;  // 45도를 라디안으로
    // double gravity_z_angle = 45.0 * M_PI / 180.0;  // 45도를 라디안으로
    
    // 중력 벡터 계산 (베이스 좌표계)
    // y축 45도: z축에서 y축으로 45도 회전
    // z축 45도: 추가적으로 x축 방향으로 구성요소 추가
    Eigen::Vector3d gravity_base;
    gravity_base << 0,  // x 성분
                    - tool_mass * gravity_magnitude * cos(45.0 * M_PI / 180.0),                         // y 성분  
                    tool_mass * gravity_magnitude * cos(45.0 * M_PI / 180.0);  // z 성분
    
    // TCP 좌표계로 중력 벡터 변환
    Eigen::Vector3d gravity_tcp = R_tcp.transpose() * gravity_base;
    
    // TCP 좌표계에서 툴 오프셋 위치
    Eigen::Vector3d tool_position_tcp;
    tool_position_tcp << 0.0, 0.0, tool_offset_z;  // TCP에서 z축으로 200mm
    
    // 모멘트 = r x F (외적) - TCP 좌표계에서
    Eigen::Vector3d moment_tcp = tool_position_tcp.cross(gravity_tcp);
    
    // TCP 좌표계에서 툴 중력 힘과 모멘트
    Eigen::Vector<double, 6> F_tool_gravity_tcp;
    F_tool_gravity_tcp << gravity_tcp(0), gravity_tcp(1), gravity_tcp(2), 
                          moment_tcp(0), moment_tcp(1), moment_tcp(2);
    
    // TCP 좌표계의 툴 중력을 베이스 좌표계로 변환
    // 6x6 변환 행렬 생성 (회전 행렬을 6DOF로 확장)
    Eigen::Matrix<double, 6, 6> T_tcp_to_base = Eigen::Matrix<double, 6, 6>::Zero();
    T_tcp_to_base.block<3,3>(0,0) = R_tcp;  // 힘 변환
    T_tcp_to_base.block<3,3>(3,3) = R_tcp;  // 모멘트 변환
    
    // // 베이스 좌표계에서의 툴 중력 (최종)
    // // Eigen::Vector<double, 6> F_tool_gravity = T_tcp_to_base * F_tool_gravity_tcp;
    // F_tool_gravity = T_tcp_to_base * F_tool_gravity_tcp;
    
    // 툴 무게 보상: 센서에서 측정된 힘에서 툴 무게 제거
    // F_e = F_e - F_tool_gravity;
    // F_e = F_e - F_tool_gravity;
    
    // 외력 오프셋 보정 (자동 캘리브레이션으로 대체되어 주석 처리)
    // F_e(0) += 0.0;   // X축 힘
    // F_e(1) += 10.0;   // Y축 힘
    // F_e(2) += -12.0;  // Z축 힘
    
    // === Desired Trajectory (첫 실행 시 현재 위치로 설정) ===
    // if (!x_d_initialized)
    if (!x_d_initialized && first_get)
    {
        x_d_initial = x_current;  // 첫 번째 실행 시 현재 위치를 목표 위치로 설정
        
        // 초기 자세도 quaternion으로 저장
        double a_init = x_current(3) * M_PI / 180.0;  // Z 회전 (A)
        double b_init = x_current(4) * M_PI / 180.0;  // Y 회전 (B)
        double c_init = x_current(5) * M_PI / 180.0;  // X 회전 (C)
        Eigen::Matrix3d R_init =
            (Eigen::AngleAxisd(a_init, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(b_init, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(c_init, Eigen::Vector3d::UnitX())).toRotationMatrix();
        quat_d_initial = Eigen::Quaterniond(R_init);
        quat_d_initial.normalize();
        
        x_d_initialized = true;
        printf("Initial desired position set to: [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
               x_d_initial(0), x_d_initial(1), x_d_initial(2), 
               x_d_initial(3), x_d_initial(4), x_d_initial(5));
    }
    
    // VR Command Integration (Joint Space to Task Space)
    // Check if VR command is received
    // VR Command Integration
    // 우선순위: Task Space Command > Joint Space Command > Default Position
    if (task_command_received)
    {
        // Task Space 명령이 있으면 최우선 사용
        std::lock_guard<std::mutex> lock(vr_mtx);
        x_d = x_d_vr;  // Task space 목표를 직접 사용!
        
        // 디버깅 출력 (처음 한 번만)
        static bool task_debug_printed = false;
        if (!task_debug_printed) {
            printf("[Task Command] x_d: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] (mm, deg)\n",
                   x_d(0), x_d(1), x_d(2), x_d(3), x_d(4), x_d(5));
            task_debug_printed = true;
        }
    }
    // else if (vr_command_received)
    // {
    //     // Task Space 명령이 없으면 Joint Space 명령 사용
    //     std::lock_guard<std::mutex> lock(vr_mtx);
    //     // VR에서 받은 Joint 값을 Forward Kinematics로 TCP 좌표로 변환
    //     x_d = forwardKinematics_M0609(q_d_vr);
        
    //     // 디버깅: FK 계산 결과 확인 (첫 번째 실행 시)
    //     static bool fk_debug_printed = false;
    //     if (!fk_debug_printed) {
    //         printf("[FK Debug] q_d_vr: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] rad\n",
    //                q_d_vr(0), q_d_vr(1), q_d_vr(2), q_d_vr(3), q_d_vr(4), q_d_vr(5));
    //         printf("[FK Debug] x_d:    [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] (mm, deg)\n",
    //                x_d(0), x_d(1), x_d(2), x_d(3), x_d(4), x_d(5));
    //         fk_debug_printed = true;
    //     }
    // }
    else
    {
        // 아무 명령도 없으면 기본 위치 사용
        x_d = x_d_initial;
        // x_d << -400.0, 15.0, 680.0,  15.0, 15.0, 15.0;                                // 목표 위치 (초기 위치 유지)
        // x_d << -541.635,  12.897, 402.973, 100.348, 100.762, -100.835;   
    }
    
    x_dot_d = Eigen::Vector<double, 6>::Zero();       // 목표 속도
    x_ddot_d = Eigen::Vector<double, 6>::Zero();      // 목표 가속도
    
    // === Error Calculation ===
    // 1) 위치 에러
    Eigen::Vector3d p_current = x_current.head<3>();  // [x,y,z]
    Eigen::Vector3d p_des     = x_d.head<3>();        // [x_d, y_d, z_d]

    // ZYX Euler angles (Doosan 방식)
    double a_cur = x_current(3) * M_PI / 180.0;  // 첫 번째 Z 회전 (A)
    double b_cur = x_current(4) * M_PI / 180.0;  // Y 회전 (B)
    double c_cur = x_current(5) * M_PI / 180.0;  // 두 번째 X 회전 (C)

    // ZYX 회전 행렬 구성: R = Rz(a) * Ry(b) * Rx(c)
    Eigen::Matrix3d R_current =
        (Eigen::AngleAxisd(a_cur, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(b_cur, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(c_cur, Eigen::Vector3d::UnitX())).toRotationMatrix();
    
    Eigen::Quaterniond quat_current(R_current);

    // 목표 자세 (quaternion 직접 사용)
    Eigen::Quaterniond quat_des;
    
    if (task_command_received)
    {
        // VR에서 Task Space 명령이 들어온 경우
        quat_des = quat_d_vr;
    }
    else
    {
        // VR 명령이 없으면 초기 자세 유지
        quat_des = quat_d_initial;
    }

    // ZYX Euler angles (degrees) for display
    Eigen::Vector3d euler_zyx = quat_des.toRotationMatrix().eulerAngles(2, 1, 0) * 180.0 / M_PI;
    x_d.tail<3>() = euler_zyx;
    
    // 쿼터니언 에러 계산 및 최단 경로 보정
    Eigen::Quaterniond q_err = quat_des * quat_current.conjugate();
    if(q_err.w() < 0) q_err.coeffs() *= -1;  // shortest path

    // Angle-Axis 벡터로 변환 (axis * angle)
    Eigen::AngleAxisd angle_axis(q_err);
    Eigen::Vector3d rot_error_vector = angle_axis.axis() * angle_axis.angle(); 
    // rot_error_vector는 [Rx, Ry, Rz] 순서의 회전 벡터 (라디안)

    // 6) 최종 delta_x 구성
    delta_x.head<3>() = p_des - p_current;  // 위치 에러 [x, y, z]
    delta_x.tail<3>() = rot_error_vector * 180.0 / M_PI;  // 회전 에러 [Rx, Ry, Rz] (degrees)

    // [2026-04-12] 속도 에러 계산 전 x_dot_current LPF 적용 (이동 시 거친 촉각/진동 제거)
    static Eigen::Vector<double, 6> x_dot_lpf = Eigen::Vector<double, 6>::Zero();
    static bool is_first_lpf = true;
    if (is_first_lpf) {
        x_dot_lpf = x_dot_current;
        is_first_lpf = false;
    } else {
        double alpha_v = 0.02; 
        x_dot_lpf = alpha_v * x_dot_current + (1.0 - alpha_v) * x_dot_lpf;
    }

    // 6) 속도 에러는 LPF된 속도로 계산
    delta_x_dot = x_dot_d - x_dot_lpf;

    

    // === Desired Impedance Parameters ===
    // 질량 행렬 M_d (6x6) - 대각 행렬
    M_d << m_lin[0],  0.0,  0.0,  0.0,  0.0,  0.0,   // x축
           0.0,  m_lin[1],  0.0,  0.0,  0.0,  0.0,   // y축
           0.0,  0.0,  m_lin[2],  0.0,  0.0,  0.0,   // z축
           0.0,  0.0,  0.0,  m_ang[0],  0.0,  0.0,   // rx축
           0.0,  0.0,  0.0,  0.0, m_ang[1],  0.0,   // ry축
           0.0,  0.0,  0.0,  0.0,  0.0, m_ang[2];   // rz축
    
    
    // 강성 행렬 K_d (6x6) - 대각 행렬
    K_d << k_lin[0],  0.0,  0.0,  0.0,  0.0,  0.0,   // x축
           0.0, k_lin[1],  0.0,  0.0,  0.0,  0.0,    // y축
           0.0,  0.0, k_lin[2],  0.0,  0.0,  0.0,    // z축
           0.0,  0.0,  0.0, k_ang[0],  0.0,  0.0,    // rx축
           0.0,  0.0,  0.0,  0.0, k_ang[1],  0.0,    // ry축
           0.0,  0.0,  0.0,  0.0,  0.0, k_ang[2];    // rz축

    // 댐핑 행렬 B_d (6x6) - 대각 행렬
    B_d << c_lin[0],  0.0,  0.0,  0.0,  0.0,  0.0,   // x축
           0.0, c_lin[1],  0.0,  0.0,  0.0,  0.0,    // y축
           0.0,  0.0, c_lin[2],  0.0,  0.0,  0.0,    // z축
           0.0,  0.0,  0.0, c_ang[0],  0.0,  0.0,    // rx축
           0.0,  0.0,  0.0,  0.0, c_ang[1],  0.0,    // ry축
           0.0,  0.0,  0.0,  0.0,  0.0, c_ang[2];    // rz축
    
    // 댐핑 행렬 B_d (임계 댐핑 계산 옵션)
    // B_d = Eigen::Matrix<double, 6, 6>::Zero();
    // for(int i = 0; i < 6; i++) {
    //     B_d(i, i) = 2.0 * sqrt(M_d(i, i) * K_d(i, i));
    // }

    
    // === Jacobian Derivative Calculation ===
    // 고정된 제어주기 dt, LPF 계수 alpha in (0,1)
    // J_dot = Eigen::Matrix<double, 6, 6>::Zero();
    static Eigen::Matrix<double,6,6> J_prev, J_next, Jdot_lpf = Eigen::Matrix<double,6,6>::Zero();
    static bool has_prev = false, has_next = false;
    
    // 자코비안 업데이트 및 미분 계산
    const double dt = 0.001;  // 1ms 제어 주기
    const double alpha = 0.02; // LPF 계수 (0.1 = 더 부드럽게, 0.9 = 더 반응적으로)
    
    J_prev = J_next;
    J_next = J;
    has_prev = has_next;
    has_next = true;
    
    if(has_prev && has_next) {
        Eigen::Matrix<double,6,6> Jdot = (J_next - J_prev) / dt; //(2.0 * dt);
        Jdot_lpf = alpha * Jdot + (1.0 - alpha) * Jdot_lpf;  // 간단 LPF
    }
    
    J_dot = Jdot_lpf;

    // === Tool Weight Compensation ===
    // Feedforward torque to compensate for the weight of the attached tool.
    // Uses Flange-frame gravity transformation and Jacobian transposition.
    {
        double m_tool;
        double cog_offset_z;
        std::vector<double> gravity_dir;
        this->get_parameter("tool.mass", m_tool);
        this->get_parameter("tool.offset_z", cog_offset_z);
        this->get_parameter("tool.gravity_direction", gravity_dir);

        // ① R_BF: Base → Flange rotation matrix
        //    (R_current is already computed above from actual_tcp_position ZYX Euler angles)
        const Eigen::Matrix3d& R_BF = R_current;

        // ② Gravity vector in Flange frame: ^F_g = R_BF^T * g_base
        //    tool.gravity_direction: normalized gravity direction in Base frame * 9.81
        //    e.g., standard floor mount pointing down = {0.0, 0.0, -9.81}
        Eigen::Vector3d g_base(
            gravity_dir[0] * 9.81,
            gravity_dir[1] * 9.81,
            gravity_dir[2] * 9.81);                        // [m/s²] in Base frame
        Eigen::Vector3d g_flange = R_BF.transpose() * g_base;  // ^F_g [m/s²]

        // ③ Tool Force and Torque in Flange frame
        Eigen::Vector3d F_tool_fl = m_tool * g_flange;            // F_tool = m * ^F_g  [N]
        Eigen::Vector3d p_cog_fl(0.0, 0.0, cog_offset_z);        // CoG position along Flange Z [m]
        Eigen::Vector3d T_tool_fl = p_cog_fl.cross(F_tool_fl);   // T_tool = ^F_P_CoG × F_tool  [Nm]

        // ④ Transform wrench from Flange frame to Base/World frame
        //    (Jacobian is expressed in Base frame, so wrench must also be in Base frame)
        Eigen::Vector3d F_tool_base = R_BF * F_tool_fl;
        Eigen::Vector3d T_tool_base = R_BF * T_tool_fl;

        // Assemble 6D tool wrench in Base frame [N; Nm]
        F_tool_gravity.head<3>() = F_tool_base;
        F_tool_gravity.tail<3>() = T_tool_base;
    }
    // ⑤ Joint-space feedforward torque: τ_tool = J(q)^T * [F_tool; T_tool]
    Eigen::Vector<double, 6> tau_tool_comp = J_T * F_tool_gravity;

    // === Impedance Control Law ===
    // τ = M_e(q)J^{-1}(q)[ẍ_d + M_d^{-1}{B_d Δẋ + K_d Δx - F_e} - J̇(q,q̇)q̇] + C(q,q̇) + G(q) + J^T(q)F_e
    Eigen::Vector<double, 6> B_delta_x_dot_val = B_d * delta_x_dot;
    Eigen::Vector<double, 6> K_delta_x_val = K_d * delta_x;
    Eigen::Vector<double, 6> impedance_force = M_d.inverse() * (B_delta_x_dot_val + K_delta_x_val + F_e);
    // [2026-04-12] Fix unit scale: J_dot (from Pinocchio) expects q_dot in rad/s.
    // If we use q_dot_current (deg/s), the Coriolis acceleration is artificially amplified by 57.3x.
    Eigen::Vector<double, 6> q_dot_rad;
    for(int i=0; i<6; i++) {
        q_dot_rad(i) = q_dot_current(i) * M_PI / 180.0;
    }
    Eigen::Vector<double, 6> J_dot_q_dot = J_dot * q_dot_rad;
    Eigen::Vector<double, 6> acceleration_ref = x_ddot_d + impedance_force - J_dot_q_dot;

    // Publish acceleration_ref (task-space acceleration reference)
    if (acceleration_ref_publisher_) {
        std_msgs::msg::Float64MultiArray acc_msg;
        acc_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) acc_msg.data[i] = acceleration_ref(i);
        acceleration_ref_publisher_->publish(acc_msg);
    }

    // Compute and publish J_inv * acceleration_ref
    Eigen::Vector<double, 6> Jinv_acc = J_inv * acceleration_ref;
    if (Jinv_acc_publisher_) {
        std_msgs::msg::Float64MultiArray jinv_msg;
        jinv_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) jinv_msg.data[i] = Jinv_acc(i);
        Jinv_acc_publisher_->publish(jinv_msg);
    }
    
    // Compute B_d * delta_x_dot and K_d * delta_x for publishing / debugging
    Eigen::Vector<double, 6> B_times_delta_dot = B_d * delta_x_dot;
    Eigen::Vector<double, 6> K_times_delta = K_d * delta_x;
    
    // 각 항 계산 (출력용)
    Eigen::Vector<double, 6> term1 = M_q * J_inv * acceleration_ref;    // M_e * J^-1 * acceleration_ref
    Eigen::Vector<double, 6> term2 = C_q * q_dot_current;               // C(q,q̇)
    Eigen::Vector<double, 6> term3 = G_q;                               // G(q)
    Eigen::Vector<double, 6> term4 = J_T * F_e;                         // J^T * F_e
    Eigen::Vector<double, 6> term6 = M_q.diagonal();                    // M_q의 대각 성분
    Eigen::Vector<double, 6> term7 = J_inv * acceleration_ref;          // J^-1 * acceleration_ref
    
    // Joint space torque calculation
    Eigen::Vector<double, 6> C_dot_q = C_q * q_dot_current;
    tau_impedance = M_q * J_inv * acceleration_ref + C_dot_q + G_q - J_T * F_e - tau_tool_comp;

    // 원래 계산된 토크 저장 (출력용)
    Eigen::Vector<double, 6> tau_calculated = tau_impedance;

    // === Friction Compensation ===
    // tau_f(k+1) = K * [ sum_{i=0}^k { tau_impedance(i) - tau_JTS(i) - tau_f(i) } * T - J_m * q_dot(k) ] + tau_f(0)
    // tau_impedance를 모터 명령으로 간주하고, actual_joint_torque(JTS)와의 차이를 적분하여 마찰 추정
    static Eigen::Vector<double, 6> tau_f_imp = Eigen::Vector<double, 6>::Zero();
    static Eigen::Vector<double, 6> tau_f0_imp = Eigen::Vector<double, 6>::Zero();
    static Eigen::Vector<double, 6> integral_term_imp = Eigen::Vector<double, 6>::Zero();
    static bool friction_imp_initialized = false;

    // const double K_friction_imp = 0.7;
    const double K_friction_imp = 4.0;
    const double T_sample_imp = 0.001;  // 1ms 제어 주기
    const double J_m_imp[6] = {
        0.0004956,
        0.0004956,
        0.0001839,
        0.00009901,
        0.00009901,
        0.00009901
    };

    // 관절별 관성 스케일 계수 (게인 조정용)
    const double J_scale_imp[6] = {
        800.0,   // Joint 1
        800.0,   // Joint 2
        2000.0,  // Joint 3
        2000.0,  // Joint 4
        2000.0,  // Joint 5
        400.0    // Joint 6
    };

    if (first_get)
    {
        if (!friction_imp_initialized)
        {
            for (int i = 0; i < 6; i++)
            {
                tau_f0_imp(i) = 0.0;
                tau_f_imp(i) = 0.0;
                integral_term_imp(i) = 0.0;
            }
            friction_imp_initialized = true;
        }

        // tau_joint_current: 현재 실제 관절 토크 (JTS 측정값)
        for (int i = 0; i < 6; i++)
        {
            double tau_joint_i = g_stRTState.actual_joint_torque[i];
            double q_dot_i_deg = q_dot_current(i);  // deg/s

            // 적분 항 업데이트: (명령 토크 - 실제 관절 토크 - 현재 마찰 추정) * dt
            integral_term_imp(i) += (tau_impedance(i) - tau_joint_i - tau_f_imp(i)) * T_sample_imp;

            // 마찰 추정 토크 갱신
            // tau_f = K * (integral - J_scale * J_m * q_dot) + tau_f0
            tau_f_imp(i) = K_friction_imp * (integral_term_imp(i) - J_scale_imp[i] * J_m_imp[i] * q_dot_i_deg)
                           + tau_f0_imp(i);
        }
    }

    // 마찰 보상이 적용된 최종 명령 토크
    // tau_out = tau_impedance + tau_f  (마찰로 인한 손실을 보충)
    Eigen::Vector<double, 6> tau_out = tau_impedance + tau_f_imp;
    
    // 안전 제한: G_q를 기준으로 ±torque_limit[i] 범위로 제한 (마찰 보상 포함 후 클리핑)
    for(int i = 0; i < 6; i++)
    {
        double lower_limit = G_q(i) - torque_limit[i];
        double upper_limit = G_q(i) + torque_limit[i];
        
        if (tau_out(i) < lower_limit)
        {
            tau_out(i) = lower_limit;
        }
        else if (tau_out(i) > upper_limit)
        {
            tau_out(i) = upper_limit;
        }
    }

    // tau_impedance를 클리핑된 최종값으로 갱신 (로그 및 publish 일관성)
    tau_impedance = tau_out;

    // 결과를 C 배열로 복사
    for(int i=0; i<6; i++)
    {
        trq_d[i] = tau_impedance(i);
    }

    // Publish impedance_force as Float64MultiArray on /imp_force
    std_msgs::msg::Float64MultiArray imp_msg;
    imp_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) imp_msg.data[i] = impedance_force(i);
    if (imp_force_publisher_) imp_force_publisher_->publish(imp_msg);

    // Publish delta_x (task-space position/orientation error) on /delta_x
    std_msgs::msg::Float64MultiArray delta_msg;
    delta_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) delta_msg.data[i] = delta_x(i);
    if (delta_x_publisher_) delta_x_publisher_->publish(delta_msg);

    // Publish delta_x_dot (task-space velocity error) on /delta_x_dot
    std_msgs::msg::Float64MultiArray delta_dot_msg;
    delta_dot_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) delta_dot_msg.data[i] = delta_x_dot(i);
    if (delta_x_dot_publisher_) delta_x_dot_publisher_->publish(delta_dot_msg);

    // Publish F_e (external force/torque) on /F_e
    std_msgs::msg::Float64MultiArray F_e_msg;
    F_e_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) F_e_msg.data[i] = F_e(i);
    if (F_e_publisher_) F_e_publisher_->publish(F_e_msg);

    // Publish F_e_raw on /F_e_raw
    std_msgs::msg::Float64MultiArray F_e_raw_msg;
    F_e_raw_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) F_e_raw_msg.data[i] = F_e_raw(i);
    if (F_e_raw_publisher_) F_e_raw_publisher_->publish(F_e_raw_msg);

    // Publish desired pose on /desired_pose
    std_msgs::msg::Float64MultiArray desired_msg;
    desired_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) desired_msg.data[i] = x_d(i);
    if (desired_pose_publisher_) desired_pose_publisher_->publish(desired_msg);

    // Publish current pose on /current_pose
    std_msgs::msg::Float64MultiArray cur_pose_msg;
    cur_pose_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) cur_pose_msg.data[i] = x_current(i);
    if (current_pose_publisher_) current_pose_publisher_->publish(cur_pose_msg);

    // Publish J_dot (6x6 flattened row-major) on /J_dot
    if (J_dot_publisher_) {
        std_msgs::msg::Float64MultiArray Jdot_msg;
        Jdot_msg.data.resize(36);
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                Jdot_msg.data[r*6 + c] = J_dot(r,c);
            }
        }
        J_dot_publisher_->publish(Jdot_msg);
    }

    // Publish J_dot * q_dot_current (6-element) on /Jdot_qdot
    if (Jdot_qdot_publisher_) {
        std_msgs::msg::Float64MultiArray jdot_qdot_msg;
        jdot_qdot_msg.data.resize(6);
        Eigen::Matrix<double,6,1> jdot_qdot = J_dot * q_dot_current;
        for (int i = 0; i < 6; ++i) {
            jdot_qdot_msg.data[i] = jdot_qdot(i);
        }
        Jdot_qdot_publisher_->publish(jdot_qdot_msg);
    }

    // Publish B_d * delta_x_dot on /B_delta_x_dot
    if (B_delta_x_dot_publisher_) {
        std_msgs::msg::Float64MultiArray b_msg;
        b_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) {
            b_msg.data[i] = B_times_delta_dot(i);
        }
        B_delta_x_dot_publisher_->publish(b_msg);
    }

    // Publish K_d * delta_x on /K_delta_x
    if (K_delta_x_publisher_) {
        std_msgs::msg::Float64MultiArray k_msg;
        k_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) {
            k_msg.data[i] = K_times_delta(i);
        }
        K_delta_x_publisher_->publish(k_msg);
    }

    // Publish intermediate terms term1..term4
    if (term1_publisher_) {
        std_msgs::msg::Float64MultiArray t1_msg;
        t1_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) t1_msg.data[i] = term1(i);
        term1_publisher_->publish(t1_msg);
    }
    if (term2_publisher_) {
        std_msgs::msg::Float64MultiArray t2_msg;
        t2_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) t2_msg.data[i] = term2(i);
        term2_publisher_->publish(t2_msg);
    }
    if (term3_publisher_) {
        std_msgs::msg::Float64MultiArray t3_msg;
        t3_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) t3_msg.data[i] = term3(i);
        term3_publisher_->publish(t3_msg);
    }
    if (term4_publisher_) {
        std_msgs::msg::Float64MultiArray t4_msg;
        t4_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) t4_msg.data[i] = term4(i);
        term4_publisher_->publish(t4_msg);
    }
    if (term6_publisher_) {
        std_msgs::msg::Float64MultiArray t6_msg;
        t6_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) t6_msg.data[i] = term6(i);
        term6_publisher_->publish(t6_msg);
    }
    if (term7_publisher_) {
        std_msgs::msg::Float64MultiArray t7_msg;
        t7_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) t7_msg.data[i] = term7(i);
        term7_publisher_->publish(t7_msg);
    }
    
    // <----- your control logic end -----/>

    auto message = dsr_msgs2::msg::TorqueRtStream(); 
    message.tor={trq_d[0],trq_d[1],trq_d[2],trq_d[3],trq_d[4],trq_d[5]};
    message.time=0.0;

    static int print_counter = 0;
    static const int PRINT_INTERVAL = 50; // 50ms마다 출력 (1ms * 50)

    if(first_get)
    {
        this->publisher_->publish(message);

        print_counter++;
        if(print_counter >= PRINT_INTERVAL)
        {
            print_counter = 0;

            printf("\033[2J\033[H"); // 화면 클리어 + 커서를 맨 위로
            printf("=== [VR_R MODE] Impedance Control Data ===\n");
            printf("VR Command Received: %s\n", vr_command_received.load() ? "YES" : "NO");
            // if (vr_command_received)
            // {
            //     std::lock_guard<std::mutex> lock(vr_mtx);
            //     printf("q_d_vr (rad):  [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //         q_d_vr(0), q_d_vr(1), q_d_vr(2), q_d_vr(3), q_d_vr(4), q_d_vr(5));
            // }
            
            printf("\n=== Torque Data ===\n");
            printf("G_q:           [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                G_q(0), G_q(1), G_q(2), G_q(3), G_q(4), G_q(5));
            printf("tau_calculated:[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                tau_calculated(0), tau_calculated(1), tau_calculated(2), 
                tau_calculated(3), tau_calculated(4), tau_calculated(5));
            printf("tau_limited:   [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                tau_impedance(0), tau_impedance(1), tau_impedance(2), 
                tau_impedance(3), tau_impedance(4), tau_impedance(5));
            
            printf("\n=== Impedance Control Terms ===\n");
            printf("impedance_force:      [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                impedance_force(0), impedance_force(1), impedance_force(2), 
                impedance_force(3), impedance_force(4), impedance_force(5));
            printf("acceleration_ref:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                acceleration_ref(0), acceleration_ref(1), acceleration_ref(2), 
                acceleration_ref(3), acceleration_ref(4), acceleration_ref(5));
            printf("term1(M_e*J_inv*acc): [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term1(0), term1(1), term1(2), term1(3), term1(4), term1(5));
            printf("term6(M_q):\n");
            for(int i=0; i<6; i++)
            {
                printf("  [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                    M_q(i,0), M_q(i,1), M_q(i,2), M_q(i,3), M_q(i,4), M_q(i,5));
            }
            printf("term7(J_inv*acc):     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term7(0), term7(1), term7(2), term7(3), term7(4), term7(5));
            printf("term2(C*q_dot):       [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term2(0), term2(1), term2(2), term2(3), term2(4), term2(5));
            printf("term3(G_q):           [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term3(0), term3(1), term3(2), term3(3), term3(4), term3(5));
            printf("term4(J_T*F_e):       [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term4(0), term4(1), term4(2), term4(3), term4(4), term4(5));
            printf("tau_tool_comp:        [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                tau_tool_comp(0), tau_tool_comp(1), tau_tool_comp(2), 
                tau_tool_comp(3), tau_tool_comp(4), tau_tool_comp(5));
            printf("tau_f_imp(friction):  [%8.4f][%8.4f][%8.4f][%8.4f][%8.4f][%8.4f]\n",
                tau_f_imp(0), tau_f_imp(1), tau_f_imp(2),
                tau_f_imp(3), tau_f_imp(4), tau_f_imp(5));
            printf("integral_term_imp:    [%8.4f][%8.4f][%8.4f][%8.4f][%8.4f][%8.4f]\n",
                integral_term_imp(0), integral_term_imp(1), integral_term_imp(2),
                integral_term_imp(3), integral_term_imp(4), integral_term_imp(5));
            
            printf("\n=== State Information ===\n");
            printf("trq_d:         [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                trq_d[0], trq_d[1], trq_d[2], trq_d[3], trq_d[4], trq_d[5]);
            printf("actual_jt:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                g_stRTState.actual_joint_torque[0], g_stRTState.actual_joint_torque[1], 
                g_stRTState.actual_joint_torque[2], g_stRTState.actual_joint_torque[3], 
                g_stRTState.actual_joint_torque[4], g_stRTState.actual_joint_torque[5]);
            printf("ext_JT:        [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                ext_JT[0],ext_JT[1],ext_JT[2],ext_JT[3],ext_JT[4],ext_JT[5]);
            printf("ext_tcp_raw:   [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                ext_tcp[0],ext_tcp[1],ext_tcp[2],ext_tcp[3],ext_tcp[4],ext_tcp[5]);
            // printf("F_tool_gravity:[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //     F_tool_gravity(0), F_tool_gravity(1), F_tool_gravity(2), 
            //     F_tool_gravity(3), F_tool_gravity(4), F_tool_gravity(5));
            // printf("F_e(compensated):[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //     F_e(0), F_e(1), F_e(2), F_e(3), F_e(4), F_e(5));
            
            
            
            // printf("Jacobian Matrix:\n");
            // for(int i=0; i<6; i++)
            // {
            //     printf("[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //            g_stRTState.jacobian_matrix[i][0], g_stRTState.jacobian_matrix[i][1],
            //            g_stRTState.jacobian_matrix[i][2], g_stRTState.jacobian_matrix[i][3],
            //            g_stRTState.jacobian_matrix[i][4], g_stRTState.jacobian_matrix[i][5]);
            // }

            // printf("\n=== Jacobian Inverse Matrix (J_inv) ===\n");
            // for(int i=0; i<6; i++)
            // {
            //     printf("[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //         J_inv(i,0), J_inv(i,1), J_inv(i,2), J_inv(i,3), J_inv(i,4), J_inv(i,5));
            // }
            
            printf("\n=== Mass Matrix ===\n");
            printf("Doosan Mass Matrix:\n");
            for(int i=0; i<6; i++)
            {
                printf("[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                    g_stRTState.mass_matrix[i][0], g_stRTState.mass_matrix[i][1],
                    g_stRTState.mass_matrix[i][2], g_stRTState.mass_matrix[i][3],
                    g_stRTState.mass_matrix[i][4], g_stRTState.mass_matrix[i][5]);
            }
            
            printf("    \n=== Joint Space ===\n");
            printf("q_current:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                q_current(0), q_current(1), q_current(2), q_current(3), q_current(4), q_current(5));
            printf("q_dot_current: [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                q_dot_current(0), q_dot_current(1), q_dot_current(2), q_dot_current(3), q_dot_current(4), q_dot_current(5));
            
            printf("\n=== Task Space ===\n");
            printf("x_current:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                x_current(0), x_current(1), x_current(2), x_current(3), x_current(4), x_current(5));
            printf("x_desired:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                x_d(0), x_d(1), x_d(2), x_d(3), x_d(4), x_d(5));

            // printf("rpy_current:   [%8.3f][%8.3f][%8.3f]\n", 
            //     rpy_current(0), rpy_current(1), rpy_current(2));
            // printf("rpy_desired:   [%8.3f][%8.3f][%8.3f]\n", 
            //     rpy_d(0), rpy_d(1), rpy_d(2));
            
            // printf("R_current:\n");
            // printf("  [%8.5f][%8.5f][%8.5f]\n", R_current(0,0), R_current(0,1), R_current(0,2));
            // printf("  [%8.5f][%8.5f][%8.5f]\n", R_current(1,0), R_current(1,1), R_current(1,2));
            // printf("  [%8.5f][%8.5f][%8.5f]\n", R_current(2,0), R_current(2,1), R_current(2,2));
            
            // printf("R_desired:\n");
            // printf("  [%8.5f][%8.5f][%8.5f]\n", R_d(0,0), R_d(0,1), R_d(0,2));
            // printf("  [%8.5f][%8.5f][%8.5f]\n", R_d(1,0), R_d(1,1), R_d(1,2));
            // printf("  [%8.5f][%8.5f][%8.5f]\n", R_d(2,0), R_d(2,1), R_d(2,2));

            printf("delta_x:       [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                delta_x(0), delta_x(1), delta_x(2), delta_x(3), delta_x(4), delta_x(5));
            printf("x_dot_current: [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                x_dot_current(0), x_dot_current(1), x_dot_current(2), x_dot_current(3), x_dot_current(4), x_dot_current(5));
            // printf("acceleration_ref:[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //     acceleration_ref(0), acceleration_ref(1), acceleration_ref(2), 
            //     acceleration_ref(3), acceleration_ref(4), acceleration_ref(5));
            printf("===============================\n");

            fflush(stdout);
        }

        // 데이터 로깅
        if (logging_enabled) {

            fflush(stdout);
        }
        if (logging_enabled) {
            log_counter++;
            if (log_counter >= LOG_INTERVAL) {
                log_counter = 0;
                
                // 현재 시간 (밀리초 정밀도)
                auto now = std::chrono::high_resolution_clock::now();
                auto duration = now.time_since_epoch();
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
                
                // CSV 형태로 데이터 저장
                data_log_file << std::fixed << std::setprecision(6);
                data_log_file << millis << ",";
                
                // G_q 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << G_q(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // tau_calculated 데이터  
                for(int i = 0; i < 6; i++) {
                    data_log_file << tau_calculated(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // tau_limited 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << tau_impedance(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // external_joint_torque 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << ext_JT[i];
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                std::lock_guard<std::mutex> lock(vr_mtx);
                for(int i = 0; i < 6; i++) {
                    data_log_file << q_d_vr(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // acceleration_ref 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << acceleration_ref(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // M_e * J^-1 * acceleration_ref 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << term1(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // C(q,q̇) 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << term2(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // G(q) 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << term3(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";
                
                // J^T * F_e 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << term4(i);
                    data_log_file << ",";
                }
                
                // desired_pose 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << x_d(i);
                    data_log_file << ",";
                }
                
                // current_pose 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << x_current(i);
                    data_log_file << ",";
                }
                
                // F_e_raw 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << F_e_raw(i);
                    data_log_file << ",";
                }
                
                // F_e_lpf 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << F_e_lpf(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";

                // J_dot * q_dot 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << J_dot_q_dot(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";

                // B * delta_x_dot 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << B_delta_x_dot_val(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";

                // K * delta_x 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << K_delta_x_val(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";

                // x_dot_current (TCP velocity) 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << x_dot_current(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << ",";

                // q_dot_current (joint velocity) 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << q_dot_current(i);
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << std::endl;
                data_log_file.flush();
            }
        }
    }
}

void ServojRtNode::ServojRtStreamPublisher()
{
    // </----- your control logic start ----->

    // float64[6] pos               # position  
    // float64[6] vel               # velocity
    // float64[6] acc               # acceleration
    // float64    time              # time

    // <----- your control logic end -----/>
    
    auto message = dsr_msgs2::msg::ServojRtStream(); 
    message.pos={pos_d[0],pos_d[1],pos_d[2],pos_d[3],pos_d[4],pos_d[5]};
    message.vel={vel_d[0],vel_d[1],vel_d[2],vel_d[3],vel_d[4],vel_d[5]};
    message.acc={acc_d[0],acc_d[1],acc_d[2],acc_d[3],acc_d[4],acc_d[5]};
    message.time=time_d;

    if(first_get)
    {
        this->publisher_->publish(message);
        RCLCPP_INFO(this->get_logger(), "ServojRtStream Published");
    }
}

void ServolRtNode::ServolRtStreamPublisher()
{
    // </----- your control logic start ----->

    // float64[6] pos               # position  
    // float64[6] vel               # velocity
    // float64[6] acc               # acceleration
    // float64    time              # time

    // <----- your control logic end -----/>

    auto message = dsr_msgs2::msg::ServolRtStream(); 
    message.pos={pos_d[0],pos_d[1],pos_d[2],pos_d[3],pos_d[4],pos_d[5]};
    message.vel={vel_d[0],vel_d[1],vel_d[2],vel_d[3],vel_d[4],vel_d[5]};
    message.acc={acc_d[0],acc_d[1],acc_d[2],acc_d[3],acc_d[4],acc_d[5]};
    message.time=time_d;

    if(first_get)
    {
        this->publisher_->publish(message);
        RCLCPP_INFO(this->get_logger(), "ServolRtStream Published");
    }
}

int main(int argc, char **argv)
{
    // --------------------cpu affinity set-------------------- //

    // Pin the main thread to CPU 3
    // int cpu_id = std::thread::hardware_concurrency()-1;
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(cpu_id, &cpuset);
    // Pin the main thread to CPU 3 //

    // Pin the main thread to CPUs 2 and 3
    uint32_t cpu_bit_mask = 0b1100;
    cpu_set_t cpuset;
    uint32_t cpu_cnt = 0U;
    CPU_ZERO(&cpuset);
    while (cpu_bit_mask > 0U) 
    {
        if ((cpu_bit_mask & 0x1U) > 0) 
        {
            CPU_SET(cpu_cnt, &cpuset);
        }
        cpu_bit_mask = (cpu_bit_mask >> 1U);
        cpu_cnt++;
    }
    auto ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (ret>0)
    {
        std::cerr << "Couldn't set CPU affinity. Error code" << strerror(errno) << std::endl;
        return EXIT_FAILURE;
    }
    ret = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (ret<0)
    {
        std::cerr << "Coudln't get CPU affinity. Error code" << strerror(errno) << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Pinned CPUs:"<< std::endl;
    for (int i=0; i < CPU_SETSIZE; i++)
    {
        if(CPU_ISSET(i,&cpuset))
        {
            std::cout << "  CPU" << std::to_string(i) << std::endl;
        }
    }
    // Pin the main thread to CPUs 2 and 3 //

    // -------------------- cpu affinity set    -------------------- //

    // -------------------- get process scheduling option -------------------- //
    auto options_reader = SchedOptionsReader();
    if (!options_reader.read_options(argc, argv)) 
    {
        options_reader.print_usage();
        return 0;
    }
    auto options = options_reader.get_options();
    // -------------------- get process scheduling option -------------------- //

    // -------------------- middleware thread scheduling -------------------- //
    set_thread_scheduling(pthread_self(), options.policy, options.priority);
    rclcpp::init(argc,argv);

    auto node1= std::make_shared<ReadDataRtNode>();
    rclcpp::executors::SingleThreadedExecutor executor1;
    executor1.add_node(node1);
    auto executor1_thread = std::thread([&](){executor1.spin();});

    auto node2= std::make_shared<TorqueRtNode>();
    rclcpp::executors::SingleThreadedExecutor executor2;
    executor2.add_node(node2);
    auto executor2_thread = std::thread([&](){executor2.spin();});

    // VR Joint Command Subscriber Node
    auto node_vr = std::make_shared<VRJointCommandNode>();
    rclcpp::executors::SingleThreadedExecutor executor_vr;
    executor_vr.add_node(node_vr);
    auto executor_vr_thread = std::thread([&](){executor_vr.spin();});

    // Task Space Command Subscriber Node
    auto node_task = std::make_shared<TaskSpaceCommandNode>();
    rclcpp::executors::SingleThreadedExecutor executor_task;
    executor_task.add_node(node_task);
    auto executor_task_thread = std::thread([&](){executor_task.spin();});

    // CSV Pose Command Subscriber Node
    auto node_csv_pose = std::make_shared<CsvPoseCommandNode>();
    rclcpp::executors::SingleThreadedExecutor executor_csv_pose;
    executor_csv_pose.add_node(node_csv_pose);
    auto executor_csv_pose_thread = std::thread([&](){executor_csv_pose.spin();});
    
    executor1_thread.join();
    executor2_thread.join();
    executor_vr_thread.join();
    executor_task_thread.join();
    executor_csv_pose_thread.join();
    
    rclcpp::shutdown();
    return 0;
}

// ----------scheduling command example----------//
// $ ros2 run dsr_realtime_control dsr_realtime_control_vr --sched SCHED_FIFO --priority 80
// $ ros2 run dsr_realtime_control dsr_realtime_control_vr --sched SCHED_RR --priority 80
// $ ps -C dsr_realtime_control_vr -L -o tid,comm,rtprio,cls,psr
// ----------scheduling command example----------//
