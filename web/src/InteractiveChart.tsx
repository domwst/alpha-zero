import { useEffect, useRef, useState } from 'preact/hooks';
import type { JSX } from 'preact';

export function usePlotWidth() {
  const ref = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(620);
  useEffect(() => {
    const observer = new ResizeObserver(entries => setWidth(Math.max(180, Math.round(entries[0]!.contentRect.width))));
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}

export function pointerX(event: JSX.TargetedPointerEvent<SVGSVGElement> | JSX.TargetedMouseEvent<SVGSVGElement>): number {
  const svg = event.currentTarget, matrix = svg.getScreenCTM();
  if (!matrix) return 0;
  return new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()).x;
}

export function inspectKey(event: JSX.TargetedKeyboardEvent<SVGSVGElement>, index: number, count: number, onChange: (index: number) => void) {
  const next = { ArrowLeft: index - 1, ArrowRight: index + 1, Home: 0, End: count - 1 }[event.key];
  if (next === undefined) return;
  event.preventDefault();
  onChange(Math.max(0, Math.min(count - 1, next)));
}
