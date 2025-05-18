# 使用官方的 Python 3.8 镜像作为基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录下所有文件到容器的 /app 目录
COPY . /app

# 安装系统依赖（如果需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（假设你有 requirements.txt）
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 运行时默认命令，可以替换成你的主脚本
CMD ["python", "NBT.py"]
