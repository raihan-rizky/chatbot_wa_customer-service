-- ============================================
-- Products Table — Toko Teladan Customer Service
-- ============================================
-- Menyimpan katalog produk untuk referensi harga
-- Nama tabel disesuaikan dengan database POS: pos_products

CREATE TABLE IF NOT EXISTS pos_products (
    sku         VARCHAR(100) PRIMARY KEY, -- SKU / Kode Barang
    name        VARCHAR(255) NOT NULL,    -- Nama Barang
    unit        VARCHAR(50),              -- Satuan (pcs, m2, etc)
    price       NUMERIC(15, 2) NOT NULL,  -- Harga Jual
    categoryId  VARCHAR(100),             -- Kategori (Outdoor, Indoor, ATK, etc)
    material    VARCHAR(100),             -- Jenis Bahan
    stock       INTEGER DEFAULT 0,        -- Stok Tersedia
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookup
CREATE INDEX IF NOT EXISTS idx_pos_products_category ON pos_products (categoryId);

-- Insert initial sample data
INSERT INTO pos_products (sku, name, unit, price, categoryId, material, stock) VALUES
-- Outdoor Materials
('SPF-280', 'Cetak Spanduk Flexi 280gr', 'm2', 25000, 'Outdoor', 'Flexi China', 999),
('SPF-340', 'Cetak Spanduk Flexi 340gr', 'm2', 45000, 'Outdoor', 'Flexi Korea', 999),
('SPF-510', 'Cetak Spanduk Flexi 510gr', 'm2', 85000, 'Outdoor', 'Flexi Jerman', 999),

-- Indoor / High Quality Materials
('SPA', 'Cetak Albatros', 'm2', 105000, 'Indoor', 'Albatros', 999),
('SP-PVC', 'Cetak PVC Rigid', 'm2', 120000, 'Indoor', 'PVC Rigid', 999),
('SP-LUS', 'Cetak Luster', 'm2', 115000, 'Indoor', 'Luster', 999),
('ST-VIN', 'Cetak Stiker Vinyl', 'm2', 75000, 'Stiker', 'Vinyl Glossy/Doft', 999),
('ST-ONE', 'Cetak Stiker One Way', 'm2', 85000, 'Stiker', 'One Way Vision', 999),

-- ATK (Alat Tulis Kantor)
('PUL-S', 'Pulpen Standar', 'pcs', 3000, 'ATK', 'Plastic', 100),
('BUK-T', 'Buku Tulis', 'pcs', 5000, 'ATK', 'Paper', 50)

ON CONFLICT (sku) DO UPDATE 
SET name = EXCLUDED.name, 
    price = EXCLUDED.price, 
    categoryId = EXCLUDED.categoryId,
    material = EXCLUDED.material,
    stock = EXCLUDED.stock;