-- ============================================
-- Web Push + Closing Events for WAHA Webhook
-- ============================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS push_subscriptions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID,
    endpoint    TEXT NOT NULL UNIQUE,
    p256dh      TEXT NOT NULL,
    auth        TEXT NOT NULL,
    user_agent  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user_id
    ON push_subscriptions (user_id);

CREATE TABLE IF NOT EXISTS negotiations (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id               TEXT NOT NULL UNIQUE,
    customer_name         TEXT,
    last_offer_price      NUMERIC,
    status                TEXT NOT NULL DEFAULT 'open'
                          CHECK (status IN ('open', 'waiting_confirmation', 'closed', 'cancelled')),
    last_offer_message_id TEXT,
    closed_at             TIMESTAMPTZ,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_negotiations_status
    ON negotiations (status);

CREATE TABLE IF NOT EXISTS waha_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_name  TEXT NOT NULL,
    chat_id     TEXT,
    message_id  TEXT,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_waha_events_chat_id_created_at
    ON waha_events (chat_id, created_at DESC);

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_push_subscriptions_updated_at ON push_subscriptions;
CREATE TRIGGER trg_push_subscriptions_updated_at
BEFORE UPDATE ON push_subscriptions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_negotiations_updated_at ON negotiations;
CREATE TRIGGER trg_negotiations_updated_at
BEFORE UPDATE ON negotiations
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

ALTER TABLE push_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE negotiations ENABLE ROW LEVEL SECURITY;
ALTER TABLE waha_events ENABLE ROW LEVEL SECURITY;
