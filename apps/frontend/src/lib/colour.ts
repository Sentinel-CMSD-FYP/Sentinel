// A perceptually ordered dark-to-bright ramp (viridis stops), so density reads in order.
const STOPS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
]

/** Colour for a value already scaled to 0..1; out-of-range values are clamped. */
export function rampColour(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(x))
  const f = x - i
  const a = STOPS[i]
  const b = STOPS[i + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ]
}

export function rgba([r, g, b]: [number, number, number], alpha: number): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

/** CSS gradient matching the ramp, for the legend. */
export const RAMP_GRADIENT = `linear-gradient(to right, ${STOPS.map(
  ([r, g, b]) => `rgb(${r}, ${g}, ${b})`,
).join(', ')})`

/** A stable colour per track id, spread around the hue wheel. */
export function trackColour(id: number): string {
  return `hsl(${(id * 47) % 360}, 85%, 60%)`
}
