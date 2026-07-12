太完美了！看到输出 `10.16.1.11`，这意味着你已经成功跨越了最容易踩坑的“环境配置”阶段。系统的 C++ 底层库和 Python 上层接口现在已经实现了完美的版本对齐。

为了方便你未来查阅，或者在其他服务器上复现这套环境，我为你整理了一份针对 **Ubuntu 22.04 + RTX 4090 单卡 + Conda 虚拟环境 (sam6d)** 的完整标准操作流程（SOP）。

你可以直接将以下内容复制保存到你的项目文档（如 Notion 或 README.md）中：

---

# 🚀 TensorRT & trtexec 完整安装与配置指南

**适用环境**：Ubuntu 22.04 | NVIDIA RTX 4090 | Conda 虚拟环境

### 核心原理提示：

安装过程分为两步：首先在 Ubuntu 系统底层安装 TensorRT 的 C++ 库（包含 `trtexec` 工具），然后将 Conda 虚拟环境中的 Python `tensorrt` 库升级至与系统底层**完全一致的版本**，以避免序列化冲突。

---

### Phase 1: 安装系统级 TensorRT 依赖与 `trtexec` 工具

**1. 下载本地仓库安装包 (deb)**
获取匹配当前系统的 TensorRT 基础环境包：

```bash
wget https://developer.download.nvidia.com/compute/tensorrt/10.13.0/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.13.0-cuda-12.9_1.0-1_amd64.deb

```

**2. 注册本地仓库**
将下载的 deb 包安装到系统中，使其成为一个本地软件源：

```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.13.0-cuda-12.9_1.0-1_amd64.deb

```

**3. 添加 GPG 密钥**
系统信任该本地源必需的一步（注意替换为你终端实际提示的文件名）：

```bash
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.13.0-cuda-12.9/nv-tensorrt-local-4B10D0AF-keyring.gpg /usr/share/keyrings/

```

**4. 正式安装 TensorRT**
更新源列表并执行安装。*(注：如果你的系统连接了外网 NVIDIA 源，`apt` 可能会自动拉取最新的匹配版本，例如 10.16.1.11。请记录下安装时的真实版本号。)*

```bash
sudo apt-get update
sudo apt-get install tensorrt

```

**5. 配置全局环境变量（软链接）**
将 `trtexec` 链接到系统路径，方便全局任意位置调用：

```bash
sudo ln -s /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec

```

---

### Phase 2: 同步 Python 虚拟环境版本

*(在执行以下命令前，请确保已经激活了目标环境，例如 `conda activate sam6d`)*

**1. 彻底清理旧版本/冲突版本**
卸载可能存在的、基于旧版 CUDA（如 cu11）编译的废弃包，防止架构冲突：

```bash
pip uninstall tensorrt_cu11 tensorrt_cu11_bindings tensorrt_cu11_libs -y

```

**2. 安装匹配的 Python 版本**
必须安装与 Phase 1 中 `apt` 最终安装版本**完全一致**的包（精确到小版本号）：

```bash
pip install tensorrt==10.16.1.11

```

**3. 最终验证**
确认 Python 端与 C++ 端均输出相同版本号：

```bash
# 验证 Python 端
python -c "import tensorrt as trt; print('TensorRT Version:', trt.__version__)"

# 验证 C++ 端 (命令行工具)
trtexec --help

```

---

### 💡 附录：RTX 4090 极速模型转换命令

当你准备转换 ONNX 模型时，建议使用以下命令榨干 4090 的性能：

```bash
trtexec \
  --onnx=your_model.onnx \
  --saveEngine=your_model_fp16.engine \
  --fp16 \
  --workspace=8192

```

* **`--fp16`**: 启用半精度计算，在极低精度损失下让 4090 推理速度翻倍。
* **`--workspace=8192`**: 给予 TensorRT 8GB 的显存寻优空间，帮助其编译出最高效的运行图。