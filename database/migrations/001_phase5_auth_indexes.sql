-- ==============================================================================
-- Agrotech: Plant Disease Detection and Smart Recommendation System
-- Migration 001: Phase 5 Authentication & History Optimization
-- Adds performance indexes for user prediction history and audit tracking.
-- Safe, additive migration preserving all existing users and predictions.
-- ==============================================================================

USE plant_disease_db;

-- 1. Ensure composite index on predictions(user_id, created_at DESC) for fast dashboard pagination
CREATE INDEX IF NOT EXISTS idx_predictions_user_created ON predictions(user_id, created_at DESC);

-- 2. Optional: Index on predictions image_path for quick lookup
CREATE INDEX IF NOT EXISTS idx_predictions_image_path ON predictions(image_path);
