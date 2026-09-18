ARG VLLM_IMAGE=vllm-sm75:standard-image-required
ARG NODE_IMAGE=node:22.18.0-bookworm-slim@sha256:0d130e2ee18e88e1561375276daced6bff032539200173f2daf48c2e33f38ff5
FROM ${NODE_IMAGE} AS harness
WORKDIR /opt/harness
COPY harness/package.json harness/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM ${VLLM_IMAGE}
ARG SOURCE_REVISION=source-archive
LABEL org.opencontainers.image.title="VLLM-SM75 ultra" org.opencontainers.image.revision="${SOURCE_REVISION}"
COPY --from=harness /usr/local/bin/node /usr/local/bin/node
COPY --from=harness /opt/harness/node_modules /opt/harness/node_modules
COPY source /opt/sm75-workbench
COPY overlay/dsh-theme/ /opt/harness/node_modules/@deepseek-ai/dsh-client-ui-theme/lib/
COPY overlay/dsh-layout/ /opt/harness/node_modules/@deepseek-ai/dsh-client-ui-layout/lib/
COPY overlay/dsh-chat/ /opt/harness/node_modules/@deepseek-ai/dsh-client-ui-chat/lib/
COPY overlay/dsh-sidebar-right/ /opt/harness/node_modules/@deepseek-ai/dsh-client-ui-sidebar-right/lib/
COPY overlay/dsh-preview/ /opt/harness/node_modules/@deepseek-ai/dsh-client-ui-sidebar-documentpreview/lib/
COPY overlay/dsh-deliverables/ /opt/harness/node_modules/@deepseek-ai/dsh-client-ui-deliverables/lib/
COPY overlay/font-scale.py /opt/vllm-sm75/font-scale.py
RUN python3 /opt/vllm-sm75/font-scale.py \
    && /usr/local/bin/node /opt/sm75-workbench/console/install-native-plugins.mjs
COPY overlay/monitor.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/instrumentator/monitor.py
COPY overlay/pstate-supervisor.sh /opt/vllm-sm75/pstate-supervisor.sh
RUN chmod +x /opt/vllm-sm75/pstate-supervisor.sh
RUN python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/instrumentator/monitor.py
ENV SM75_CONSOLE_HOST=0.0.0.0 SM75_CONSOLE_PORT=1615 SM75_SINGLE_CONTAINER=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /opt/sm75-workbench
ENTRYPOINT ["/usr/local/bin/node", "console/server.mjs"]
