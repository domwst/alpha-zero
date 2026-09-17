import type { ComponentChildren } from "preact";

/** Mark initial loads and historical catch-up; routine live polling stays quiet. */
export function LoadingStatus({
  pending,
  children,
}: {
  pending: boolean;
  children: ComponentChildren;
}) {
  return (
    <span className="service-loading-status" role="status">
      {pending ? children : null}
    </span>
  );
}
