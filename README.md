# NetworkGym + NetworkAgent Client for Offline Reinforcement Learning

IMPORTANT NOTE: this repository contains BOTH [NetworkGym](https://github.com/IntelLabs/networkgym) and [NetworkAgent](https://github.com/pinyaras/GMAClient) with some modifications to support offline reinforcement learning:

- Many more metrics have been included in the input state vector for RL agent training.
- The state vectors are not normalized by default.
  - Instead, all values in the state are divided by 100 to keep the range of values roughly between 0-1.
- More reward functions have been added (throughput and delay).
- (`state`, `action`, `reward`, `next_state`) data can be stored offline whether training or testing using the `"store_offline"` parameter in [nqos_split/config.json](https://github.com/hmomin/networkgym/blob/main/network_gym_client/envs/nqos_split/config.json).
- Recent work is focusing on constructing agents that can effectively learn baseline-comparable or near-optimal policies using completely offline data.

If this repository is ever out-of-sync with updates to either [NetworkGym](https://github.com/IntelLabs/networkgym) or [NetworkAgent](https://github.com/pinyaras/GMAClient), please feel free to let me know via email: [momin@ucsb.edu](mailto:momin@ucsb.edu).

---

📋 **[NetworkGym Docs Website](https://intellabs.github.io/networkgym)**

📧 **[Contact Us](mailto:netaigym@gmail.com)**

💻 **[Slack](https://join.slack.com/t/networkgym/shared_invite/zt-23c6nvd5s-1l1m5iVtDZj3LcMgVspdNg)**

The NetworkGym Client stands as a Python-centric client library created for NetworkGym, an innovative Simulation-as-a-Service framework crafted to democratize network AI research and development. This Client establishes a remote connection to the NetworkGym Server/Environment hosted on the cloud, facilitating agent training.
At present, Network Gym Client supports four environments: `nqos_split`, `qos_steer`, `network_slicing` and `rmcat`.

## 📚 Class Structure

This repository includes the network_gym_client components. The network_gym_server and network_gym_env components are hosted in our vLab machines. After cloning this repository, users can launch the network_gym_client to remotely connects to the newtork_gym_server and newtork_gym_env via the northbound interface.

- network_gym_client
  - gymnasium.env: *a customized gymnasium environment that communicates with the agent.*
  - adapter: *transform the network stats measurements to obs and reward; translate action to policy that can be applied to the network.*
  - northbound_interface: *communicates network confiugration, network stats and policy between client and network_gym server/environment.*
- agent: any gymnasium compatible agent.


## ⌛ Installation:
- Clone this repository. We currently manage two versions: a stable version updated less frequently (accessible via port 8088) and a development version with more frequent updates for new features (accessible via port 8092).
  - To switch to the stable version branch, use the following command: `git checkout mlwins-v01-8088`.
  - To switch to the development version branch, use the following command: `git checkout mlwins-v01-8092`.
- (Optional) Create a new virtual python environment.
```
python3 -m venv network_venv
source network_venv/bin/activate
```
- Install Required Libraries `pip install -r requirements.txt` or:
```
pip install gymnasium
pip install pandas
pip install pyzmq
pip install wandb
pip install tensorboard
pip install stable-baselines3
pip install tqdm
pip install rich
```
- Request access to the [vLab](https://registration.intel-research.net/) machine.

## 🔗 Setup Port Forwarding to vLab Server:

**Skip this section if you plan to deploy the client on the mlwins-v01 vlab server.** Otherwise, follow the following steps to set up port forwarding from you local machine to the stable version of vlab Server. **Change the port number from 8088 to 8092 to access dev version.**
- First, setup port forwarding from the local port 8088 to the mlwins-v01 external server port 8088 via the SSH gateway using the following command in a screen session, e.g., `screen -S port8088`.
``` 
ssh -L 8088:mlwins-v01.research.intel-research.net:8088 ssh.intel-research.net
```
- If the previous command does not work, add your user account before the `ssh.intel-research.net` as follows.
```
ssh -L 8088:mlwins-v01.research.intel-research.net:8088 [YOUR_USER_NAME]@ssh.intel-research.net
```
 - If the previous command also does not work, add the following instructions to your ssh configure file, replace **[YOUR_USER_NAME]** with your user name and update **[PATH_TO_SSH]** accordingly.
```
# COMMAND: ssh mlwins

Host gateway
  HostName ssh.intel-research.net
  User [YOUR_USER_NAME]
  Port 22
  IdentityFile /home/[PATH_TO_SSH]/.ssh/id_rsa

Host mlwins
  HostName mlwins-v01.research.intel-research.net
  User [YOUR_USER_NAME]
  Port 22
  IdentityFile /home/[PATH_TO_SSH]/.ssh/id_rsa
  ProxyJump gateway
  LocalForward 8088 localhost:8088
```

## 🚀 Start NetworkGym Client:
- Start the demo client using the following command:
```
python3 start_client_demo.py
```
- When the program terminates, visualize the output using the returned WanDB website.

## 📁 File Structure:

```
📦 NetworkGym
┣ 📜 start_client_demo.py
┗ 📂 network_gym_client
  ┣ 📜 adapter.py (➡️ WanDB)
  ┣ 📜 common_config.json
  ┣ 📜 env.py
  ┣ 📜 northbound_interface.py (➡️ network_gym_server and network_gym_env)
  ┗ 📂 envs
    ┗ 📂 [ENV_NAME]
      ┣ 📜 adapter.py
      ┗ 📜 config.json
```

- Excuting the 📜 start_client_demo.py file will start a new simulation. To change the environment, modify the `env_name` parameter. The 📜 common_config.json is used in all environments. Depends on the selected environments, the 📜 config.json and 📜 adapter.py in the [ENV_NAME] folder will be loaded. The 📜 adapter.py helps preparing observations, rewards and actions for the selected environment.
- The 📜 start_client_demo.py create a Network Gym environment, which remotely connects to the ns-3 based Network Gym Simualtor (hosted in vLab machine) using the 📜 northbound_interface. 📜 start_client_demo.py also uses random samples from the action space to interact with the Network Gym environment. The results are synced to ➡️ WanDB database. We provide the following code snippet from the 📜 start_client_demo.py as an example:

```python
#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : start_client_demo.py

from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv

client_id = 0
env_name = "nqos_split"
config_json = load_config_file(env_name)
config_json["rl_config"]["agent"] = "random"
# Create the environment
env = NetworkGymEnv(client_id, config_json) # make a network env using pass client id and configure file arguements.

num_steps = 1000
obs, info = env.reset()

for step in range(num_steps):

    action = env.action_space.sample()  # agent policy that uses the observation and info
    obs, reward, terminated, truncated, info = env.step(action)

    # If the environment is end, exit
    if terminated:
        break

    # If the epsiode is up (environment still running), then start another one
    if truncated:
        obs, info = env.reset()
```


## 🔧 Configuration:
- Update the common configuration in [common_config.json](network_gym_client/common_config.json):

```json
{
  "connect_via_server_ip_and_server_port": true, //set to ture to use the server_ip and server_port to connect to internal server (this method requires Intel VPN); set to false to use the local_fowarded_port to connect to external server (this method requires port forwarding!).
  "server_ip": "gmasim-v01.jf.intel.com", //do not change (for internal users only).
  "server_port": 8092, //set to 8088 to access stable version or 8092 to access dev version.
  "local_fowarded_port": 8092, // the local port that used to forward to the external server.
  "session_name": "test",//Make sure to change the "session_name" to your assgined session name. Cannot use '-' in the name! Test account is for testing only (shared by every one). Contact us to apply for an account. 
  "session_key": "test",//Make sure to change the "session_key" to your assgined keys.
}
```

- Update the environment dependent configuration file, e.g., [network_gym_client/envs/nqos_split/config.json](network_gym_client/envs/nqos_split/config.json).
  - View configuration suggestions for arguments at [NetworkGym Docs Website](https://intellabs.github.io/networkgym/environments/mx_traffic_management/mx_traffic_splitting.html#arguments).




