export { 
  isComplexPrompt, 
  pruneText, 
  detectKnowledgeBoundary, 
  evaluateComplexityScore, 
  classifySpecialistRole,
  SpecialistRole,
  Message, 
  ClassifierOptions 
} from './classifier.js';

export { 
  CascadeRouter, 
  RouterConfig, 
  ModelConfig, 
  CascadeResponse,
  ChatOptions,
  ChatJsonOptions,
  UsageMetrics,
  RouterMetrics,
  TelemetryEvent,
  CascadeTriggeredError,
  MultiSpecialistRouterOptions,
  createMultiSpecialistRouter
} from './cascade.js';

