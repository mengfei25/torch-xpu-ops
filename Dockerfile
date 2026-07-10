FROM pytorch/manylinux2_28-builder:cpu-v2.13.0-rc1

ENV http_proxy=http://proxy.ims.intel.com:911
ENV https_proxy=http://proxy.ims.intel.com:911
ENV no_proxy=localhost,laurentides.igk.intel.com

# Disable SSL verification globally for dnf
RUN echo "sslverify=False" >> /etc/dnf/dnf.conf

# Option 1: Add Intel GPU repo and install intel-omix-devel
RUN dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --add-repo https://laurentides.igk.intel.com/repositories/rhel/8.10/intel-omix/6.0/unified/intel-gpu-8.10.repo && \
    dnf install -y intel-omix-devel && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Reinstall oneAPI DLE
RUN wget -q --no-proxy --no-check-certificate -O oneapi-dle.sh http://10.239.157.137/pytorch/xpu/intel-deep-learning-essentials-2026.1.1.11_offline.sh
RUN rm -rf /opt/intel/oneapi
RUN bash ./oneapi-dle.sh -a -s --action remove --eula accept || true
RUN bash ./oneapi-dle.sh -a -s --action install --eula accept && \
    bash ./oneapi-dle.sh -a -s --action repair --eula accept && \
    rm -rf ./oneapi-dle.sh && ls /opt/intel/oneapi
RUN source /opt/intel/oneapi/setvars.sh && icpx --version && sycl-ls

# Option 2: Download first and copy it and install intel-omix-devel
# COPY intel-gpu-8.10.repo /etc/yum.repos.d/intel-gpu-8.10.repo
# RUN dnf install -y intel-omix-devel && \
#     dnf clean all && \
#     rm -rf /var/cache/dnf

# Install cmake 3.28.4 via pip
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir cmake==3.28.4

# Extract static libs for embedding
RUN cd /opt/_internal && tar -xJf static-libs-for-embedding-only.tar.xz
