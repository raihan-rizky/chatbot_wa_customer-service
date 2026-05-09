# Closing Buyer UX

## Understanding Summary

- Buyer chat UX is focused on messages such as deal, order, lanjut, DP, and lunas.
- The bot must not redirect the buyer to another WhatsApp number.
- The bot should confirm briefly that the order/deal was received.
- The bot should tell the buyer that an admin will continue in the same chat.
- The bot should only ask for missing order details when the context is unclear.
- Relevant missing details are product, quantity or size, deadline, and name/contact when needed.
- Backend push notification to admin remains the operational handoff trigger.

## Assumptions

- The bot may still reply automatically after closing, but the reply must be short.
- Admin/operator takes over after receiving the push notification.
- Unclear context means the recent chat does not identify the item or quantity/size well enough.
- No admin dashboard changes are included in this project.

## Decision Log

- Chosen: Context-aware handoff response.
- Alternatives considered: always ask minimal details, immediate admin handoff only.
- Reason: it avoids repeated questions when the buyer already gave enough context, while still collecting useful missing information when the order is vague.

## Final Behavior

When the buyer indicates intent to close:

- If order context is clear, reply with a short confirmation and say admin will continue in this chat.
- If order context is incomplete, reply with a short confirmation, say admin will continue in this chat, and ask only for missing details.
- Do not redirect to another number.
- Do not push payment aggressively unless the buyer explicitly asks about payment.
