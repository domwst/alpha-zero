FROM alpine:3.23

ARG DASHBOARD_UID=1000
ARG DASHBOARD_GID=1000
RUN apk add --no-cache openssh-client curl \
    && addgroup -g "$DASHBOARD_GID" dashboard \
    && adduser -D -u "$DASHBOARD_UID" -G dashboard dashboard

USER dashboard
ENTRYPOINT ["ssh"]
