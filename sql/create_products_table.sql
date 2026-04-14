-- ============================================
-- Products Table — Toko Teladan Customer Service
-- ============================================
-- Menyimpan katalog produk untuk referensi harga

CREATE TABLE IF NOT EXISTS products_teladan (
    code VARCHAR(50) PRIMARY KEY,     -- Kode Barang (Unique), e.g. "SP", "BN", "ST"
    name VARCHAR(255) NOT NULL,       -- Nama Barang, e.g. "Spanduk Flexi 280gr"
    unit VARCHAR(50),                 -- Satuan, e.g. "m2", "pcs", "lbr"
    price NUMERIC(15, 2) NOT NULL,    -- Harga Satuan Default
    category VARCHAR(50),             -- Kategori, e.g. "Outdoor", "Indoor"
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookup by code
CREATE INDEX IF NOT EXISTS idx_products_code ON products_teladan (code);

-- Insert initial sample data (Banner & Large Format)
INSERT INTO products_teladan (code, name, unit, price, category) VALUES
-- Outdoor Materials
('SPF-280', 'Cetak Spanduk Flexi 280gr (China)', 'm2', 25000, 'Outdoor'),
('SPF-340', 'Cetak Spanduk Flexi 340gr (Korea)', 'm2', 45000, 'Outdoor'),
('SPF-510', 'Cetak Spanduk Flexi 510gr (Jerman)', 'm2', 85000, 'Outdoor'),

-- Indoor / High Quality Materials
('SPA', 'Cetak Albatros', 'm2', 105000, 'Indoor'),
('SP-PVC', 'Cetak PVC Rigid', 'm2', 120000, 'Indoor'),
('SP-LUS', 'Cetak Luster', 'm2', 115000, 'Indoor'),
('ST-VIN', 'Cetak Stiker Vinyl', 'm2', 75000, 'Stiker'),
('ST-ONE', 'Cetak Stiker One Way Vision', 'm2', 85000, 'Stiker'),

-- ATK (Alat Tulis Kantor)
('PUL-S', 'Pulpen Standar', 'pcs', 3000, 'ATK'),
('BUK-T', 'Buku Tulis', 'pcs', 5000, 'ATK')

ON CONFLICT (code) DO UPDATE 
SET name = EXCLUDED.name, 
    price = EXCLUDED.price, 
    category = EXCLUDED.category;