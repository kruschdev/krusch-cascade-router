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
  CrossRouterOptions,
  createCrossRouter
} from './cascade.js';

