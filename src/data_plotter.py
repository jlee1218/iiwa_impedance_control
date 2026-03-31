import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message

from geometry_msgs.msg import Pose, Wrench
from lbr_fri_idl.msg import LBRState

from scipy.spatial.transform import Rotation as R


# =========================
# File chooser
# =========================
def choose_bag():
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select ROS2 Bag Folder")
    return folder_selected


# =========================
# Quaternion → RPY
# =========================
def quat_to_rpy(q):
    r = R.from_quat([q.x, q.y, q.z, q.w])
    return r.as_euler('xyz', degrees=False)


# =========================
# Read bag
# =========================
def read_bag(bag_path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    data = {
        "cmd_pose": [], "cmd_pose_t": [],
        "meas_pose": [], "meas_pose_t": [],
        "cmd_wrench": [], "cmd_wrench_t": [],
        "meas_wrench": [], "meas_wrench_t": [],
        "joint_cmd": [], "joint_meas": [], "joint_t": [],
        "torque_cmd": [], "torque_meas": [],
        "external_torque": []
    }

    while reader.has_next():
        topic, raw, t = reader.read_next()
        t_sec = t * 1e-9

        if topic == "/impedance_controller/commanded_pose":
            msg = deserialize_message(raw, Pose)
            data["cmd_pose"].append(msg)
            data["cmd_pose_t"].append(t_sec)

        elif topic == "/impedance_controller/measured_pose":
            msg = deserialize_message(raw, Pose)
            data["meas_pose"].append(msg)
            data["meas_pose_t"].append(t_sec)

        elif topic == "/impedance_controller/commanded_wrench":
            msg = deserialize_message(raw, Wrench)
            data["cmd_wrench"].append(msg)
            data["cmd_wrench_t"].append(t_sec)

        elif topic == "/impedance_controller/measured_wrench":
            msg = deserialize_message(raw, Wrench)
            data["meas_wrench"].append(msg)
            data["meas_wrench_t"].append(t_sec)

        elif topic == "/lbr/lbr_state":
            msg = deserialize_message(raw, LBRState)

            data["joint_meas"].append(msg.measured_joint_position)
            data["joint_cmd"].append(msg.commanded_joint_position)
            data["torque_meas"].append(msg.measured_torque)
            data["torque_cmd"].append(msg.commanded_torque)
            data["external_torque"].append(msg.external_torque)

            data["joint_t"].append(t_sec)
            
    return data


# =========================
# Extract pose
# =========================
def extract_pose(pose_list):
    pose = []
    for p in pose_list:
        pose.append([
            p.position.x,
            p.position.y,
            p.position.z,
            *quat_to_rpy(p.orientation)
        ])

    return np.array(pose)


# =========================
# Extract wrench
# =========================
def extract_wrench(wrench_list):
    out = []
    for w in wrench_list:
        out.append([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z
        ])
    return np.array(out)


def interp_signal(t_old, signal, t_new):
    signal = np.array(signal)

    return np.array([
        np.interp(t_new, t_old, signal[:, i])
        for i in range(signal.shape[1])
    ]).T


def get_common_time(data):
    t_min = max([
        min(data["cmd_pose_t"]),
        min(data["meas_pose_t"]),
        min(data["cmd_wrench_t"]),
        min(data["meas_wrench_t"]),
        min(data["joint_t"])
    ])

    t_max = min([
        max(data["cmd_pose_t"]),
        max(data["meas_pose_t"]),
        max(data["cmd_wrench_t"]),
        max(data["meas_wrench_t"]),
        max(data["joint_t"])
    ])

    return np.linspace(t_min, t_max, 2000)  # resolution adjustable

# =========================
# Plot helpers
# =========================
def plot_pose(cmd_pose, meas_pose, t):
    labels = ["X [m]", "Y [m]", "Z [m]", "Roll [rads]", "Pitch [rads]", "Yaw [rads]"]
    fig, axs = plt.subplots(3, 2, sharex=True)
    fig.suptitle("Pose")

    err = cmd_pose - meas_pose

    for i in range(6):
        row, col = i % 3, i // 3

        axs[row, col].plot(t[:len(cmd_pose)], cmd_pose[:, i], label="Commanded")
        axs[row, col].plot(t[:len(meas_pose)], meas_pose[:, i], label="Measured")
        axs[row, col].plot(t[:len(err)], err[:, i], linestyle='dotted', label="Error")
        axs[row, col].set_ylabel(labels[i])
        axs[row, col].grid()

    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")
    axs[0, 0].legend()


def plot_wrench(cmd, meas, t):
    labels = ["Force_x [N]", "Force_y [N]", "Force_z [N]", "Torque_x [Nm]", "Torque_y [Nm]", "Torque_z [Nm]"]
    fig, axs = plt.subplots(3, 2, sharex=True)
    fig.suptitle("Wrench")

    cmd = np.array(cmd)
    meas = np.array(meas)
    err = cmd - meas

    for i in range(6):
        ros, col = i % 3, i // 3

        axs[ros, col].plot(t[:len(cmd)], cmd[:, i], label="Commanded")
        axs[ros, col].plot(t[:len(meas)], meas[:, i], label="Measured")
        axs[ros, col].plot(t[:len(err)], err[:, i], linestyle='dotted', label="Error")
        axs[ros, col].set_ylabel(labels[i])
        axs[ros, col].grid()

    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")
    axs[0, 0].legend()


def plot_joints(cmd, meas, t):
    fig, axs = plt.subplots(7, 1, sharex=True)
    fig.suptitle("Joint Positions")

    cmd = np.array(cmd)
    meas = np.array(meas)
    err = cmd - meas

    for i in range(7):
        axs[i].plot(t[:len(cmd)], cmd[:, i], label="Commanded")
        axs[i].plot(t[:len(meas)], meas[:, i], label="Measured")
        axs[i].plot(t[:len(err)], err[:, i], linestyle='dotted', label="Error")
        axs[i].set_ylabel(f"J{i+1}")
        axs[i].grid()

    axs[-1].set_xlabel("Time (s)")
    axs[0].legend()


def plot_external_torque(ext, t):
    fig, axs = plt.subplots(7, 1, sharex=True)
    fig.suptitle("External Torque")

    ext = np.array(ext)

    for i in range(7):
        axs[i].plot(t[:len(ext)], ext[:, i])
        axs[i].set_ylabel(f"J{i+1}")
        axs[i].grid()

    axs[-1].set_xlabel("Time (s)")


# =========================
# Main
# =========================
def main():
    bag_path = choose_bag()
    data = read_bag(bag_path)

    # Extract raw signals
    cmd_pose = extract_pose(data["cmd_pose"])
    meas_pose = extract_pose(data["meas_pose"])

    cmd_wrench = extract_wrench(data["cmd_wrench"])
    meas_wrench = extract_wrench(data["meas_wrench"])

    joint_cmd = np.array(data["joint_cmd"])
    joint_meas = np.array(data["joint_meas"])
    external_torque = np.array(data["external_torque"])

    # Get common time base
    t_common = get_common_time(data)

    # Interpolate everything
    cmd_pose_i = interp_signal(data["cmd_pose_t"], cmd_pose, t_common)
    meas_pose_i = interp_signal(data["meas_pose_t"], meas_pose, t_common)

    cmd_wrench_i = interp_signal(data["cmd_wrench_t"], cmd_wrench, t_common)
    meas_wrench_i = interp_signal(data["meas_wrench_t"], meas_wrench, t_common)

    joint_cmd_i = interp_signal(data["joint_t"], joint_cmd, t_common)
    joint_meas_i = interp_signal(data["joint_t"], joint_meas, t_common)

    ext_torque_i = interp_signal(data["joint_t"], external_torque, t_common)

    # Plot using synchronized data
    plot_pose(cmd_pose_i, meas_pose_i, t_common)
    plot_wrench(cmd_wrench_i, meas_wrench_i, t_common)
    plot_joints(joint_cmd_i, joint_meas_i, t_common)
    plot_external_torque(ext_torque_i, t_common)

    plt.show()


if __name__ == "__main__":
    main()