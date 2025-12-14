# Boris (Telegram) — Assistente Financeiro MVP

## O que faz
- Lança despesas/receitas a partir de mensagens no Telegram
- Classifica por categoria e centro de custo (heurísticas simples)
- Salva tudo no Supabase (Postgres)
- Comandos:
  - `/start` — registra seu usuário (fica pendente até ser autorizado)
  - `/autorizar <tg_user_id> role=owner|partner|buyer|viewer` — autoriza um usuário (apenas owner)
  - `/despesa <texto>` — lança uma despesa (ou apenas mande texto natural)
  - `/receita <texto>` — lança uma receita
  - `/relatorio` — resumo do mês por categoria e por centro de custo

## Variáveis de ambiente
- `TELEGRAM_TOKEN` — token do BotFather
- `SUPABASE_URL` — Project URL (https://xxxx.supabase.co)
- `SUPABASE_KEY` — anon public key do projeto

## Deploy rápido no Render
1. Crie um novo Web Service e aponte para este código (suba via Git ou uploade o zip).
2. Configure as Environment Variables acima.
3. Após o serviço ficar de pé, copie a URL pública (ex.: `https://boris-xxxxx.onrender.com`).
4. Ligue o webhook do Telegram acessando:
   `https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook?url=<URL_PUBLICA>/webhook`
5. No Telegram:
   - Envie `/start` para o bot
   - Pegue seu `tg_user_id` (o bot mostra)
   - Autorize usuários com `/autorizar <id> role=owner|partner|buyer`
6. Teste:
   - Mande: `paguei 200 no eletricista do Bloco E`
   - Use `/relatorio` para ver o resumo do mês
