// Service Worker for U'n'Eye Web Labeler:
// 1. Full offline caching for all assets (.html, .js, .wasm, .bin, .json, .css)
// 2. coi-serviceworker functionality: adds COOP/COEP headers to enable Cross-Origin Isolation (SharedArrayBuffer)

const CACHE_NAME = 'uneye-cache-v1';
const ASSETS_TO_CACHE = [
    './',
    './index.html',
    './style.css',
    './app.js',
    './formats.js',
    './train_worker.js',
    './web/uneye_train.js',
    './web/uneye_train.wasm',
    './models/combined.bin',
    './models/combined.json',
    './logo.jpeg'
];

self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(ASSETS_TO_CACHE).catch(err => {
                console.warn('Cache addAll warning:', err);
            });
        })
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        Promise.all([
            self.clients.claim(),
            caches.keys().then((keys) => {
                return Promise.all(
                    keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
                );
            })
        ])
    );
});

self.addEventListener('fetch', (event) => {
    const request = event.request;
    if (request.method !== 'GET') return;

    event.respondWith(
        caches.match(request).then(async (cachedResponse) => {
            if (cachedResponse) {
                // Add COOP / COEP headers to cached responses
                return addCoopCoepHeaders(cachedResponse);
            }

            try {
                const networkResponse = await fetch(request);
                if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
                    const responseToCache = networkResponse.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(request, responseToCache);
                    });
                }
                return addCoopCoepHeaders(networkResponse);
            } catch (err) {
                // Fallback to cache if offline
                const fallback = await caches.match(request);
                if (fallback) return addCoopCoepHeaders(fallback);
                throw err;
            }
        })
    );
});

function addCoopCoepHeaders(response) {
    if (!response) return response;
    // Opaque responses or redirected responses cannot have new headers set
    if (response.status === 0 || response.type === 'opaque') return response;

    const newHeaders = new Headers(response.headers);
    newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');
    newHeaders.set('Cross-Origin-Embedder-Policy', 'require-corp');

    return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders
    });
}
