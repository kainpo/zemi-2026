FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# 必要なツールのインストール (curl等)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# コンテナ内への uv のインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 作業ディレクトリの設定
WORKDIR /app

# プログラムファイル等のコピー
COPY . .

# パッケージのインストールはコンテナ実行時に行うため、ここでは対話シェルをデフォルトにする
CMD ["/bin/bash"]
