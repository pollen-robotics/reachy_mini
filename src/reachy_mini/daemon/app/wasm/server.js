// server.js
const express = require('express');
const path = require('path');

const app = express();

// Inject COOP/COEP headers
app.use((req, res, next) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');

    next();
});

// Static serve
app.use(express.static(path.join(process.cwd(), 'dist')));

// SPA fallback
app.get('*', (req, res) => {
    res.sendFile(path.join(process.cwd(), 'dist', 'index.html'));
});

const port = process.env.PORT || 7860;
console.log(`Listening on port ${port}`);
app.listen(port);
