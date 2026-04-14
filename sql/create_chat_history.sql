-- ============================================
-- Chat History Table — Teladan AI WhatsApp Bot
-- ============================================
-- Table name must match the code in chat_history.py: "chat_messages_teladan"

CREATE TABLE IF NOT EXISTS chat_messages_teladan (
    id            BIGSERIAL PRIMARY KEY,
    phone         VARCHAR(20) NOT NULL,              -- nomor WA user (e.g. 6281991029210)
    role          VARCHAR(10) NOT NULL                -- 'user' atau 'assistant'
                  CHECK (role IN ('user', 'assistant')),
    content       TEXT NOT NULL,                      -- isi pesan
    image_url     TEXT,                               -- optional: media ID or URL for image messages
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Index untuk pencarian riwayat chat per nomor
CREATE INDEX IF NOT EXISTS idx_chat_messages_teladan_phone
    ON chat_messages_teladan (phone, created_at DESC);

-- Opsional: RLS (Row Level Security) untuk Supabase
ALTER TABLE chat_messages_teladan ENABLE ROW LEVEL SECURITY;
