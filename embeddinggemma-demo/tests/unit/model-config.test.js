import assert from 'node:assert/strict';
import { test } from 'node:test';

import { MODELS, onnxFile, textOnlyConfig } from '../../public/lib/model-config.js';

test('each model and dtype points at its ONNX file', () => {
  assert.equal(onnxFile('v1', 'fp32'), 'onnx/model.onnx');
  assert.equal(onnxFile('v1', 'q4'), 'onnx/model_no_gather_q4.onnx');
  assert.equal(onnxFile('v2', 'q8'), 'onnx/model_quantized.onnx');
  assert.equal(onnxFile('v2', 'q4'), 'onnx/model_q4.onnx');
});

test('only fp32 of EmbeddingGemma 2 runs on WASM, and no model offers float16', () => {
  const wasm = (model) => Object.keys(MODELS[model].dtypes).filter((dtype) => !MODELS[model].dtypes[dtype].webgpuOnly);
  assert.deepEqual(wasm('v1'), ['fp32', 'q8', 'q4']);
  assert.deepEqual(wasm('v2'), ['fp32']);
  for (const { dtypes } of Object.values(MODELS)) {
    assert.ok(Object.keys(dtypes).every((dtype) => !dtype.includes('16')));
  }
});

test('EmbeddingGemma 2 loads its text model only', async () => {
  const requests = [];
  const AutoConfig = {
    from_pretrained: async (id, options) => {
      requests.push({ id, revision: options.revision });
      return { model_type: 'embedding_gemma2', vision_config: {}, audio_config: {}, text_config: {} };
    },
  };
  assert.equal(await textOnlyConfig(AutoConfig, 'v1'), undefined);
  const config = await textOnlyConfig(AutoConfig, 'v2');
  assert.equal(config.vision_config, null);
  assert.equal(config.audio_config, null);
  assert.deepEqual(config.text_config, {});
  assert.deepEqual(requests, [{ id: MODELS.v2.id, revision: MODELS.v2.revision }]);
});
