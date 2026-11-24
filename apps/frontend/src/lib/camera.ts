import type { Camera } from '../types'

/** Upload form fields for the clip's camera; empty strings are left out of the request. */
export interface CameraFields {
  altitude_m: string
  fov_deg: string
  fov_kind: 'horizontal' | 'diagonal'
  sensor_aspect: string
  camera_model: string
  altitude_source: Camera['altitude_source']
}

export const EMPTY_CAMERA: CameraFields = {
  altitude_m: '',
  fov_deg: '',
  fov_kind: 'horizontal',
  sensor_aspect: '4:3',
  camera_model: '',
  altitude_source: 'assumed',
}

/** Only the fields worth sending: none at all when altitude and FOV are both empty. */
export function cameraForm(f: CameraFields): Record<string, string> {
  if (!f.altitude_m.trim() && !f.fov_deg.trim()) return {}
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(f)) if (v.trim()) out[k] = v.trim()
  if (f.fov_kind !== 'diagonal') delete out.sensor_aspect
  return out
}

/** One line for the clip's info row, e.g. "60 m (assumed), HFOV 71.5°, DJI Mini 3". */
export function cameraLabel(c: Camera): string {
  const parts = [`${c.altitude_m} m (${c.altitude_source})`, `HFOV ${Math.round(c.hfov_deg * 10) / 10}°`]
  if (c.model) parts.push(c.model)
  return parts.join(', ')
}
