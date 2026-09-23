ARG VLLM_IMAGE=vllm-sm75:standard-image-required
ARG NODE_IMAGE=node:22.23.2-bookworm-slim@sha256:43aeff40f4afc22e83f7589a2f37e111cff5ca84529571f1c8415bcc5fcc21b2
FROM ${NODE_IMAGE} AS harness
WORKDIR /opt/harness
COPY harness/package.json harness/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM ${VLLM_IMAGE}
ARG SOURCE_REVISION=source-archive
LABEL org.opencontainers.image.title="VLLM-SM75 Ultra" org.opencontainers.image.revision="${SOURCE_REVISION}"
COPY --from=harness /usr/local/bin/node /usr/local/bin/node
COPY --from=harness /opt/harness/node_modules /opt/harness/node_modules
RUN ln -sf /opt/harness/node_modules/pnpm/bin/pnpm.mjs /usr/local/bin/pnpm \
    && pnpm --version
COPY source /opt/sm75-workbench
ARG INSTALL_LMCACHE=0
RUN if [ "${INSTALL_LMCACHE}" = "1" ]; then python3 /opt/sm75-workbench/lmcache/install.py --install --build-no-gpu; fi
COPY overlay/font-scale.py /opt/vllm-sm75/font-scale.py
RUN python3 /opt/vllm-sm75/font-scale.py \
    && /usr/local/bin/node /opt/sm75-workbench/console/install-native-plugins.mjs \
    && /usr/local/bin/node /opt/sm75-workbench/console/verify-client-bundles.mjs
COPY overlay/pstate-supervisor.sh /opt/vllm-sm75/pstate-supervisor.sh
RUN chmod +x /opt/vllm-sm75/pstate-supervisor.sh
# Monitor APIs are inherited from the matching standard image.
ENV SM75_CONSOLE_HOST=0.0.0.0 SM75_CONSOLE_PORT=1615 SM75_SINGLE_CONTAINER=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /opt/sm75-workbench
ENTRYPOINT ["/usr/local/bin/node", "console/server.mjs"]
