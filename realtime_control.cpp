#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "dsr_realtime_control/realtime_control.hpp"

#include <pthread.h>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <cerrno>

RT_STATE g_stRTState;
std::mutex mtx;
std::atomic_bool first_get(false);

// 데이터 로깅용 파일 스트림
std::ofstream data_log_file;
std::atomic_bool logging_enabled(false);
static int log_counter = 0;
static const int LOG_INTERVAL = 10; // 10ms마다 로깅 (1ms * 10)

using namespace DRAFramework;
CDRFLEx Drfl;

ReadDataRtNode::ReadDataRtNode() : Node("ReadDataRt")
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

TorqueRtNode::TorqueRtNode() : Node("TorqueRt")
{
    publisher_  = this->create_publisher<dsr_msgs2::msg::TorqueRtStream>("/dsr01/torque_rt_stream",10);
    timer_      = this->create_wall_timer(std::chrono::microseconds(1000),std::bind(&TorqueRtNode::TorqueRtStreamPublisher,this));

    // 파라미터 선언 및 기본값 설정
    this->declare_parameter("impedance.mass.linear", std::vector<double>{20.0, 20.0, 20.0});
    this->declare_parameter("impedance.mass.angular", std::vector<double>{20.0, 20.0, 20.0});
    this->declare_parameter("impedance.damping.linear", std::vector<double>{0.01, 0.01, 0.01});
    this->declare_parameter("impedance.damping.angular", std::vector<double>{0.01, 0.01, 0.01});
    this->declare_parameter("impedance.stiffness.linear", std::vector<double>{10.0, 10.0, 10.0});
    this->declare_parameter("impedance.stiffness.angular", std::vector<double>{10.0, 10.0, 10.0});
    
    this->declare_parameter("control.torque_limit", 25.0);
    this->declare_parameter("control.print_interval", 50);
    this->declare_parameter("control.log_interval", 10);
    this->declare_parameter("control.jacobian_lpf_alpha", 0.1);
    
    this->declare_parameter("tool.mass", 1.5);
    this->declare_parameter("tool.offset_z", 0.2);
    this->declare_parameter("tool.gravity_direction", std::vector<double>{0.0, -0.707, 0.707});
    
    this->declare_parameter("desired_position.position", std::vector<double>{-450.0, 50.0, 450.0});
    this->declare_parameter("desired_position.orientation", std::vector<double>{0.0, -60.0, -135.0});

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
    filename << data_dir << "/impedance_control_data_" 
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
        data_log_file << "J_T_F_e_0,J_T_F_e_1,J_T_F_e_2,J_T_F_e_3,J_T_F_e_4,J_T_F_e_5";
        data_log_file << std::endl;
        logging_enabled = true;
        RCLCPP_INFO(this->get_logger(), "Data logging started: %s", filename.str().c_str());
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to open log file: %s", filename.str().c_str());
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

ServojRtNode::ServojRtNode() : Node("ServojRt")
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

ServolRtNode::ServolRtNode() : Node("ServolRt")
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
                    g_stRTState.mass_matrix[i][j] = response->data.mass_matrix[i].data[j];
                    g_stRTState.jacobian_matrix[i][j] = response->data.jacobian_matrix[i].data[j];
                }
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
    
    for(int i=0; i<6; i++)
    {
        mtx.lock();

        // 현재 관절 상태
        q_current(i) = g_stRTState.actual_joint_position[i];
        q_dot_current(i) = g_stRTState.actual_joint_velocity[i];
        
        // 외부 힘/토크 (원시 데이터)
        // F_e(i) = g_stRTState.external_tcp_force[i];
        // 0,1,2는 음수로, 3,4,5는 양수로 받음
        if (i < 3) {
            F_e(i) = -g_stRTState.external_tcp_force[i];
        } 
        // else if (i > 5)
        // {
        //     F_e(i) = -g_stRTState.external_tcp_force[i];
        // }
        else {
            F_e(i) = g_stRTState.external_tcp_force[i];
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
    
    // === Compute Derived Quantities ===
    J_T = J.transpose();
    J_inv = J.inverse();  // 또는 pseudoInverse() 사용
    
    // End-effector mass matrix: M_e = (J * M_q^-1 * J^T)^-1
    // M_e = (J * M_q.inverse() * J_T).inverse();

    // === Current TCP State (Forward Kinematics 필요) ===
    // C 배열을 Eigen 벡터로 복사
    for(int i = 0; i < 6; i++)
    {
        x_current(i) = g_stRTState.actual_tcp_position[i];
    }
    x_dot_current = J * q_dot_current;                 // TCP 속도
    
    // === Tool Weight Compensation ===
    // 툴 무게: 1.5kg, TCP에서 z축으로 200mm 앞에 위치
    double tool_mass = 1.5;  // kg
    double tool_offset_z = 0.2;  // 200mm = 0.2m
    double gravity_magnitude = 9.81;   // m/s²
    
    // TCP 자세 (rx, ry, rz) 추출
    double rx = x_current(3);  // roll (x축 회전)
    double ry = x_current(4);  // pitch (y축 회전) 
    double rz = x_current(5);  // yaw (z축 회전)
    
    // 회전 행렬 생성 (ZYX Euler angle 순서)
    Eigen::Matrix3d R_z, R_y, R_x;
    R_z << cos(rz), -sin(rz), 0,
           sin(rz),  cos(rz), 0,
           0,        0,       1;
    
    R_y << cos(ry),  0, sin(ry),
           0,        1, 0,
           -sin(ry), 0, cos(ry);
    
    R_x << 1, 0,        0,
           0, cos(rx), -sin(rx),
           0, sin(rx),  cos(rx);
    
    // 최종 회전 행렬: R = R_z * R_y * R_x
    Eigen::Matrix3d R_tcp = R_z * R_y * R_x;
    
    // 베이스 좌표계에서 중력 벡터 (y축 45도, z축 45도 방향)
    // y축 45도: yz 평면에서 45도 회전
    // z축 45도: xz 평면에서 45도 회전 (추가적인 회전)
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
    
    // 베이스 좌표계에서의 툴 중력
    F_tool_gravity = T_tcp_to_base * F_tool_gravity_tcp;
    
    // 툴 무게 보상: 센서에서 측정된 힘에서 툴 무게 제거
    // F_e = F_e - F_tool_gravity;
    
    // === Desired Trajectory (첫 실행 시 현재 위치로 설정) ===
    // if (!x_d_initialized && first_get)
    if (!x_d_initialized)
    {
        x_d_initial = x_current;  // 첫 번째 실행 시 현재 위치를 목표 위치로 설정
        x_d_initialized = true;
        printf("Initial desired position set to: [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
               x_d_initial(0), x_d_initial(1), x_d_initial(2), 
               x_d_initial(3), x_d_initial(4), x_d_initial(5));
    }
    
    // x_d = x_d_initial;                                // 목표 위치 (초기 위치 유지)
    x_d << -450.0,  50.0, 450.0,  150.0, -110.0, -150.0;
    x_dot_d = Eigen::Vector<double, 6>::Zero();       // 목표 속도
    x_ddot_d = Eigen::Vector<double, 6>::Zero();      // 목표 가속도
    
    // === Error Calculation ===
    delta_x = x_d - x_current;
    delta_x_dot = x_dot_d - x_dot_current;
    
    // === Desired Impedance Parameters ===
    // 질량 행렬 M_d (6x6) - 대각 행렬
    M_d << 4.0,  0.0,  0.0,  0.0,  0.0,  0.0,   // x축: 4.0 kg
           0.0,  4.0,  0.0,  0.0,  0.0,  0.0,   // y축: 4.0 kg
           0.0,  0.0,  4.0,  0.0,  0.0,  0.0,   // z축: 4.0 kg
           0.0,  0.0,  0.0,  9.0,  0.0,  0.0,   // rx축: 9.0 kg⋅m²
           0.0,  0.0,  0.0,  0.0,  9.0,  0.0,   // ry축: 9.0 kg⋅m²
           0.0,  0.0,  0.0,  0.0,  0.0,  9.0;   // rz축: 9.0 kg⋅m²
    
    
    // 강성 행렬 K_d (6x6) - 대각 행렬
    K_d << 0.8,  0.0,  0.0,  0.0,  0.0,  0.0,   // x축: 0.8 N/m
           0.0,  0.8,  0.0,  0.0,  0.0,  0.0,   // y축: 0.8 N/m
           0.0,  0.0,  0.8,  0.0,  0.0,  0.0,   // z축: 0.8 N/m
           0.0,  0.0,  0.0,  15.0, 0.0,  0.0,   // rx축: 15.0 N⋅m/rad
           0.0,  0.0,  0.0,  0.0,  15.0, 0.0,   // ry축: 15.0 N⋅m/rad
           0.0,  0.0,  0.0,  0.0,  0.0,  15.0;  // rz축: 15.0 N⋅m/rad

    // 댐핑 행렬 B_d (6x6) - 대각 행렬
    // B_d << 2.8,  0.0,  0.0,  0.0,  0.0,  0.0,  // x축: 2.8 N⋅s/m
    //        0.0, 2.8,  0.0,  0.0,  0.0,  0.0,   // y축: 2.8 N⋅s/m
    //        0.0,  0.0, 2.8,  0.0,  0.0,  0.0,   // z축: 2.8 N⋅s/m
    //        0.0,  0.0,  0.0, 14.0,  0.0,  0.0,   // rx축: 14.0 N⋅m⋅s/rad
    //        0.0,  0.0,  0.0,  0.0, 14.0,  0.0,   // ry축: 14.0 N⋅m⋅s/rad
    //        0.0,  0.0,  0.0,  0.0,  0.0, 14.0;   // rz축: 14.0 N⋅m⋅s/rad
    
    // 댐핑 행렬 B_d (6x6) - 대각 행렬 (임계 댐핑: B_d = 2*sqrt(M_d * K_d))
    B_d = Eigen::Matrix<double, 6, 6>::Zero();
    for(int i = 0; i < 6; i++) {
        B_d(i, i) = 2.0 * sqrt(M_d(i, i) * K_d(i, i));
    }


    // === Jacobian Derivative Calculation ===
    // 고정된 제어주기 dt, LPF 계수 alpha in (0,1)
    // J_dot = Eigen::Matrix<double, 6, 6>::Zero();
    static Eigen::Matrix<double,6,6> J_prev, J_next, Jdot_lpf = Eigen::Matrix<double,6,6>::Zero();
    static bool has_prev = false, has_next = false;
    
    // 자코비안 업데이트 및 미분 계산
    const double dt = 0.001;  // 1ms 제어 주기
    const double alpha = 0.05; // LPF 계수 (0.1 = 더 부드럽게, 0.9 = 더 반응적으로)
    
    J_prev = J_next;
    J_next = J;
    has_prev = has_next;
    has_next = true;
    
    if(has_prev && has_next) {
        Eigen::Matrix<double,6,6> Jdot = (J_next - J_prev) / (2.0 * dt); // 2dt? 아닌듯!!!!!! 계산해야된다고함
        Jdot_lpf = alpha * Jdot + (1.0 - alpha) * Jdot_lpf;  // 간단 LPF
    }
    
    J_dot = Jdot_lpf;

    // === Impedance Control Law ===
    // τ = M_e(q)J^{-1}(q)[ẍ_d + M_d^{-1}{B_d Δẋ + K_d Δx - F_e} - J̇(q,q̇)q̇] + C(q,q̇) + G(q) + J^T(q)F_e
    Eigen::Vector<double, 6> impedance_force = M_d.inverse() * (B_d * delta_x_dot + K_d * delta_x - F_e);
    Eigen::Vector<double, 6> acceleration_ref = x_ddot_d + impedance_force - J_dot * q_dot_current;
    
    // 각 항 계산 (출력용)
    Eigen::Vector<double, 6> term1 = M_q * J_inv * acceleration_ref;    // M_e * J^-1 * acceleration_ref
    Eigen::Vector<double, 6> term2 = C_q * q_dot_current;               // C(q,q̇)
    Eigen::Vector<double, 6> term3 = G_q;                               // G(q)
    Eigen::Vector<double, 6> term4 = J_T * F_e;                         // J^T * F_e
    
    // Joint space torque calculation
    Eigen::Vector<double, 6> C_dot_q = C_q * q_dot_current;
    tau_impedance = M_q * J_inv * acceleration_ref + C_dot_q + G_q + J_T * F_e;

    // 원래 계산된 토크 저장 (출력용)
    Eigen::Vector<double, 6> tau_calculated = tau_impedance;
    
    // 안전 제한: G_q를 기준으로 ±1 범위로 제한
    for(int i = 0; i < 6; i++)
    {
        double lower_limit = G_q(i) - 0.01;
        double upper_limit = G_q(i) + 0.01;
        
        if (tau_impedance(i) < lower_limit)
        {
            tau_impedance(i) = lower_limit;
        }
        else if (tau_impedance(i) > upper_limit)
        {
            tau_impedance(i) = upper_limit;
        }
    }

    
    // 결과를 C 배열로 복사
    for(int i=0; i<6; i++)
    {
        trq_d[i] = tau_impedance(i);
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
            printf("=== Impedance Control Data ===\n");
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
            printf("term2(C*q_dot):       [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term2(0), term2(1), term2(2), term2(3), term2(4), term2(5));
            printf("term3(G_q):           [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term3(0), term3(1), term3(2), term3(3), term3(4), term3(5));
            printf("term4(J_T*F_e):       [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                term4(0), term4(1), term4(2), term4(3), term4(4), term4(5));
            
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
            printf("F_tool_gravity:[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                F_tool_gravity(0), F_tool_gravity(1), F_tool_gravity(2), 
                F_tool_gravity(3), F_tool_gravity(4), F_tool_gravity(5));
            printf("F_e(compensated):[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                F_e(0), F_e(1), F_e(2), F_e(3), F_e(4), F_e(5));
            
            
            printf("Jacobian Matrix:\n");
            for(int i=0; i<6; i++)
            {
                printf("[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                       g_stRTState.jacobian_matrix[i][0], g_stRTState.jacobian_matrix[i][1],
                       g_stRTState.jacobian_matrix[i][2], g_stRTState.jacobian_matrix[i][3],
                       g_stRTState.jacobian_matrix[i][4], g_stRTState.jacobian_matrix[i][5]);
            }

            // printf("\n=== Jacobian Inverse Matrix (J_inv) ===\n");
            // for(int i=0; i<6; i++)
            // {
            //     printf("[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //         J_inv(i,0), J_inv(i,1), J_inv(i,2), J_inv(i,3), J_inv(i,4), J_inv(i,5));
            // }
            
            // printf("\n=== Mass Matrix ===\n");
            // printf("Mass Matrix:\n");
            // for(int i=0; i<6; i++)
            // {
            //     printf("[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
            //         g_stRTState.mass_matrix[i][0], g_stRTState.mass_matrix[i][1],
            //         g_stRTState.mass_matrix[i][2], g_stRTState.mass_matrix[i][3],
            //         g_stRTState.mass_matrix[i][4], g_stRTState.mass_matrix[i][5]);
            // }
            
            printf("\n=== Joint Space ===\n");
            printf("q_current:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                q_current(0), q_current(1), q_current(2), q_current(3), q_current(4), q_current(5));
            printf("q_dot_current: [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                q_dot_current(0), q_dot_current(1), q_dot_current(2), q_dot_current(3), q_dot_current(4), q_dot_current(5));
            
            printf("\n=== Task Space ===\n");
            printf("x_current:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                x_current(0), x_current(1), x_current(2), x_current(3), x_current(4), x_current(5));
            printf("x_desired:     [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                x_d(0), x_d(1), x_d(2), x_d(3), x_d(4), x_d(5));
            printf("delta_x:       [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                delta_x(0), delta_x(1), delta_x(2), delta_x(3), delta_x(4), delta_x(5));
            printf("x_dot_current: [%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                x_dot_current(0), x_dot_current(1), x_dot_current(2), x_dot_current(3), x_dot_current(4), x_dot_current(5));
            printf("acceleration_ref:[%8.3f][%8.3f][%8.3f][%8.3f][%8.3f][%8.3f]\n", 
                acceleration_ref(0), acceleration_ref(1), acceleration_ref(2), 
                acceleration_ref(3), acceleration_ref(4), acceleration_ref(5));
            printf("===============================\n");

            fflush(stdout);
        }

        // 데이터 로깅 (10ms마다)
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
                
                // impedance_force 데이터
                for(int i = 0; i < 6; i++) {
                    data_log_file << impedance_force(i);
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
                    if(i < 5) data_log_file << ",";
                }
                data_log_file << std::endl;
                data_log_file.flush(); // 즉시 파일에 쓰기
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

    // auto node3= std::make_shared<ServojRtNode>();
    // rclcpp::executors::SingleThreadedExecutor executor3;
    // executor3.add_node(node3);
    // auto executor3_thread = std::thread([&](){executor3.spin();});

    // auto node4= std::make_shared<ServolRtNode>();
    // rclcpp::executors::SingleThreadedExecutor executor4;
    // executor4.add_node(node4);
    // auto executor4_thread = std::thread([&](){executor4.spin();});


    // -------------------- middleware thread scheduling -------------------- //

    // -------------------- realtime executor scheduling -------------------- //

    // set_thread_scheduling(executor1_thread.native_handle(), options.policy, options.priority);
    // set_thread_scheduling(executor2_thread.native_handle(), options.policy, options.priority);
    // set_thread_scheduling(executor3_thread.native_handle(), options.policy, options.priority);
    // set_thread_scheduling(executor4_thread.native_handle(), options.policy, options.priority);
    
    // -------------------- realtime executor scheduling -------------------- //

    executor1_thread.join();
    executor2_thread.join();
    // executor3_thread.join();
    // executor4_thread.join();
    
    rclcpp::shutdown();
    return 0;
}

// ----------scheduling command example----------//
// $ ros2 run dsr_realtime_control realtime_control --sched SCHED_FIFO --priority 80
// $ ros2 run dsr_realtime_control realtime_control --sched SCHED_RR --priority 80
// $ ps -C realtime_control -L -o tid,comm,rtprio,cls,psr
// ----------scheduling command example----------//