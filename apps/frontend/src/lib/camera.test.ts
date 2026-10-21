import { describe, expect, it } from 'vitest'
import { cameraForm, cameraLabel, EMPTY_CAMERA } from './camera'

describe('cameraForm', () => {
  it('sends nothing when altitude and FOV are empty', () => {
    expect(cameraForm({ ...EMPTY_CAMERA, camera_model: 'DJI Mini 3' })).toEqual({})
  })
  it('sends the filled fields, and the sensor shape only for a diagonal FOV', () => {
    const f = { ...EMPTY_CAMERA, altitude_m: ' 60 ', fov_deg: '71.5' }
    expect(cameraForm(f)).toEqual({ altitude_m: '60', fov_deg: '71.5', fov_kind: 'horizontal', altitude_source: 'assumed' })
    expect(cameraForm({ ...f, fov_kind: 'diagonal' })).toMatchObject({ fov_kind: 'diagonal', sensor_aspect: '4:3' })
  })
  it('still sends a half filled form, so the backend can say what is missing', () => {
    expect(cameraForm({ ...EMPTY_CAMERA, altitude_m: '60' })).toMatchObject({ altitude_m: '60' })
  })
})

describe('cameraLabel', () => {
  it('names altitude, its source, the horizontal FOV and the model', () => {
    expect(cameraLabel({ altitude_m: 60, altitude_source: 'assumed', hfov_deg: 71.53 })).toBe('60 m (assumed), HFOV 71.5°')
    expect(cameraLabel({ altitude_m: 45, altitude_source: 'telemetry', hfov_deg: 69.73, model: 'DJI Mini 3' })).toBe(
      '45 m (telemetry), HFOV 69.7°, DJI Mini 3',
    )
  })
})
