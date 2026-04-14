# WhatsApp Customer Service AI — Toko Teladan

A WhatsApp Customer Service AI bot built with **FastAPI**, **LangChain**, **Nebius AI Studio** (LLM & Vision), and **WAHA** (WhatsApp HTTP API).

This bot acts as a digital customer service assistant for "Toko Teladan Percetakan dan ATK". It answers customer inquiries, provides product catalogs and pricing, estimates printing costs, and analyzes design images sent by customers.

## Features

- **Conversational AI Customer Service**: Powered by `Qwen 3 14B` via Nebius, providing natural, helpful, and polite responses in Indonesian.
- **Product Catalog Knowledge**: Built-in knowledge of banner, sticker, and stationery prices.
- **Design Image Analysis**: Customers can send pictures of their designs (logos, sketches, etc.) and the bot will use `Qwen 2.5 Vision` to analyze it and suggest suitable printing materials and estimates.
- **Persistent Memory**: Uses Supabase to store chat history, allowing the bot to remember context per customer.

## Prerequisites

1. **Python 3.10+**
2. **Nebius API Key** (for LLM and Vision)
3. **Supabase Project** (for chat memory)
4. **WAHA** running (Docker or WAHA Cloud)

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Copy the example and fill in your keys:
   ```bash
   cp .env.example .env
   ```

3. **Supabase Database**:
   Execute the sql file provided (`sql/create_chat_history.sql`) in your Supabase SQL Editor.

## Run Locally

Start the FastAPI application:

```bash
uvicorn app.main:app --reload --port 8000
```

## How It Works

1. Customers message the WhatsApp number connected to WAHA.
2. WAHA sends a webhook payload to `/webhook`.
3. The bot retrieves the customer's previous chat history from Supabase.
4. The AI (System Prompt) acts as an admin for Toko Teladan, answering questions and guiding orders.
5. If an image is sent, the bot passes it to the Vision model to analyze the design and provide recommendations.
6. The AI response is sent back to WAHA to be delivered to the customer.

## API Endpoints

| Method | Path       | Description                  |
| ------ | ---------- | ---------------------------- |
| `GET`  | `/`        | Health check                 |
| `POST` | `/webhook` | Receive incoming WA messages |
