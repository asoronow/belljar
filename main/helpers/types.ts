// types.ts

export interface AnimalMetadata {
  hasCellDetectionData: boolean;
  hasAlignmentData: boolean;
  cellDetectionRun: boolean;
  alignmentRun: boolean;
}

export interface ProjectMetadata {
  name: string;
  createdAt: string;
  lastModified: string;
  description: string;
  animals: Record<string, AnimalMetadata>;
}
