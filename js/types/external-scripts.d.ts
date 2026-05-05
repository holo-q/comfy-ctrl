/**
 * Ambient type definitions for external ComfyUI scripts
 *
 * These modules are provided by ComfyUI at runtime and are not part of this package.
 * This file provides TypeScript declarations so imports can be resolved during compilation.
 */

// Wildcard module declaration for app.js at any depth
declare module '**/scripts/app.js' {
  import type { ComfyApp } from './types/comfyui';
  export const app: ComfyApp;
}

// Wildcard module declaration for api.js at any depth
declare module '**/scripts/api.js' {
  import type { ComfyApi } from './types/comfyui';
  export const api: ComfyApi;
}
