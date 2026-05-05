/**
 * Type definitions for ComfyUI core scripts
 * These are external dependencies loaded by ComfyUI
 */

import type {
  ComfyApp,
  ComfyApi,
  LiteGraph as LiteGraphType
} from './comfyui';

/**
 * ComfyUI app instance
 * Available at: ../../scripts/app.js (from root level imports)
 */
declare module '../../scripts/app.js' {
  export const app: ComfyApp;
}

/**
 * ComfyUI app instance (from nested directories)
 * Available at: ../../../scripts/app.js
 */
declare module '../../../scripts/app.js' {
  export const app: ComfyApp;
}

/**
 * ComfyUI API instance
 * Available at: ../../scripts/api.js (from root level imports)
 */
declare module '../../scripts/api.js' {
  export const api: ComfyApi;
}

/**
 * ComfyUI API instance (from nested directories)
 * Available at: ../../../scripts/api.js
 */
declare module '../../../scripts/api.js' {
  export const api: ComfyApi;
}

/**
 * LiteGraph global
 * Available as global variable in ComfyUI browser environment
 */
declare global {
  const LiteGraph: LiteGraphType;

  interface Window {
    LiteGraph: LiteGraphType;
  }
}

// Export empty object to make this a module (required for declare module to work)
export {};
