-- À exécuter dans Supabase (SQL Editor) si la table feedback a des colonnes en CHAR(1).
-- Corrige les types pour accepter des textes (prédiction, diagnostic, commentaire).

ALTER TABLE feedback
  ALTER COLUMN predicted_class TYPE text,
  ALTER COLUMN diagnostic TYPE text,
  ALTER COLUMN comment TYPE text;

-- Si la colonne image_url existe et est en CHAR(1) :
-- ALTER TABLE feedback ALTER COLUMN image_url TYPE text;
