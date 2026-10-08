import assert from 'node:assert/strict';
import { test } from 'node:test';

import { PHONE_MODELS, assessLiteRt, assessVariant, findVariant, variantKey } from '../../public/lib/phone-models.js';

const GiB = 1024 ** 3;
const phoneGpu = (overrides = {}) => ({ f16: true, maxBufferSize: 2 * GiB, maxStorageBufferBindingSize: 2 * GiB, ...overrides });
const phone = (webgpu, extra = {}) => ({ mobile: true, deviceMemoryGB: null, webgpu, ...extra });

test('every variant key round-trips through findVariant', () => {
  for (const model of PHONE_MODELS) {
    for (const variant of model.variants) {
      const found = findVariant(variantKey(model, variant));
      assert.equal(found.model, model);
      assert.equal(found.variant, variant);
    }
  }
  assert.equal(findVariant('gemma-3-270m:q4:wasm'), null);
});

test('WebGPU variants need an adapter, f16 when quantized to q4f16, and room for the largest weight', () => {
  const gemma4 = findVariant('gemma-4-e2b:q4f16:webgpu').variant;
  assert.equal(assessVariant(gemma4, phone(null)).level, 'no');
  assert.equal(assessVariant(gemma4, phone(phoneGpu({ f16: false }))).level, 'no');
  // The 1174 MB per-layer embedding table does not fit a 1 GiB buffer limit.
  const small = assessVariant(gemma4, phone(phoneGpu({ maxStorageBufferBindingSize: GiB })));
  assert.equal(small.level, 'no');
  assert.match(small.reasons.join(), /1,174 MB が GPU の 1 バッファの上限 1,073 MB/);
  // With room for it, the 3.1 GB download still makes it borderline.
  assert.equal(assessVariant(gemma4, phone(phoneGpu())).level, 'maybe');

  const gemma270 = findVariant('gemma-3-270m:q4f16:webgpu').variant;
  const defaultLimits = phoneGpu({ maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2 });
  assert.deepEqual(assessVariant(gemma270, phone(defaultLimits)), { level: 'yes', reasons: [] });
  // EmbeddingGemma 2's 8-bit embedding table is exactly 128 MiB, so it fits WebGPU's default limits too.
  assert.deepEqual(assessVariant(findVariant('embeddinggemma-2:q8:webgpu').variant, phone({ ...defaultLimits, f16: false })), { level: 'yes', reasons: [] });
  assert.equal(assessVariant(findVariant('embeddinggemma-2:q4:webgpu').variant, phone(null)).level, 'no');
});

test('WASM variants are judged by size, and phones get a warning for large ones', () => {
  const fp32 = findVariant('gemma-3-270m:fp32:wasm').variant;
  assert.equal(assessVariant(fp32, { mobile: false, deviceMemoryGB: null, webgpu: null }).level, 'yes');
  assert.equal(assessVariant(fp32, phone(null)).level, 'maybe');
  assert.equal(assessVariant(findVariant('embeddinggemma:q4:wasm').variant, phone(null)).level, 'yes');
  assert.equal(assessVariant(fp32, phone(null, { deviceMemoryGB: 2 })).level, 'maybe');
});

test('the LiteRT-LM build needs f16 and about 500 MB per buffer', () => {
  assert.equal(assessLiteRt(phone(null)).level, 'no');
  assert.equal(assessLiteRt(phone(phoneGpu({ f16: false }))).level, 'no');
  assert.equal(assessLiteRt(phone(phoneGpu({ maxStorageBufferBindingSize: 256 * 1024 ** 2 }))).level, 'no');
  assert.equal(assessLiteRt(phone(phoneGpu())).level, 'maybe');
});
