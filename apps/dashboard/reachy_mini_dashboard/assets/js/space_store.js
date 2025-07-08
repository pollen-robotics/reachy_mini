let spacesStore = null;

// Initialize spaces store on page load
function initializeSpacesStore() {
    spacesStore = new SpacesStore();
    spacesStore.init();
}

function refreshSpacesStore() {
    if (spacesStore) {
        spacesStore.renderSpaces();
    }
}

// HF Spaces Store functionality
class SpacesStore {
    constructor() {
        this.spaces = [];
        this.filteredSpaces = [];
        this.currentSort = 'likes';
        this.searchTerm = '';
    }

    async init() {
        await this.loadSpaces();
        this.setupEventListeners();
        this.renderSpaces();
    }

    async loadSpaces() {
        try {
            // Search for spaces with the reachy_mini tag
            const response = await fetch('https://huggingface.co/api/spaces?filter=reachy_mini&sort=likes&direction=-1&limit=50&full=true');
            const data = await response.json();

            this.spaces = data.map(space => ({
                author: space.author,
                created: new Date(space.createdAt).getTime(),
                // TODO: Keep the full URL for the space
                // But at the moment, only the last part is kept by the installed packages
                id: space.id.split('/').pop(), // Extract the space ID from the URL
                cardData: space.cardData || {},
                likes: space.likes || 0,
                // url: `https://huggingface.co/spaces/${space.id}`,
                installUrl: `https://huggingface.co/spaces/${space.id}`,
                // tags: space.tags || []
            }));

            this.filteredSpaces = [...this.spaces];
            this.updateStats();
        } catch (error) {
            this.showError();
        }
    }

    setupEventListeners() {
        const searchInput = document.getElementById('spaces-search');
        searchInput.addEventListener('input', (e) => {
            this.searchTerm = e.target.value.toLowerCase();
            this.filterSpaces();
        });

        const sortButtons = document.querySelectorAll('.sort-btn');
        sortButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                sortButtons.forEach(b => {
                    b.classList.remove('active', 'bg-blue-500', 'text-white');
                    b.classList.add('bg-gray-200', 'text-gray-700');
                });
                e.target.classList.remove('bg-gray-200', 'text-gray-700');
                e.target.classList.add('active', 'bg-blue-500', 'text-white');
                this.currentSort = e.target.dataset.sort;
                this.sortSpaces();
            });
        });
    }

    filterSpaces() {
        this.filteredSpaces = this.spaces.filter(space =>
            space.title.toLowerCase().includes(this.searchTerm) ||
            space.author.toLowerCase().includes(this.searchTerm) ||
            space.description.toLowerCase().includes(this.searchTerm)
        );
        this.sortSpaces();
    }

    sortSpaces() {
        switch (this.currentSort) {
            case 'likes':
                this.filteredSpaces.sort((a, b) => b.likes - a.likes);
                break;
            case 'created':
                this.filteredSpaces.sort((a, b) => b.created - a.created);
                break;
            case 'name':
                this.filteredSpaces.sort((a, b) => a.title.localeCompare(b.title));
                break;
        }
        this.renderSpaces();
    }

    updateStats() {
        const statsEl = document.getElementById('spaces-stats');
        // const total = this.spaces.length;
        // const totalLikes = this.spaces.reduce((sum, space) => sum + space.likes, 0);
        // statsEl.innerHTML = `Found ${total} spaces with ${totalLikes.toLocaleString()} total likes`;
        statsEl.innerHTML = '';
    }

    renderSpaceCard(space, dashboardStatus) {
        const title = space.cardData?.title + ' ' + space.cardData?.emoji || space.id;
        const description = space.cardData?.short_description || '';

        const colorFromName = (space.cardData?.colorFrom || "yellow").toLowerCase();
        const colorToName = (space.cardData?.colorTo || "orange").toLowerCase();
        const colorGradient = `bg-gradient-to-br from-${colorFromName}-700 to-${colorToName}-700`;

        return `
            <a class="space-card ${colorGradient}" href="${space.installUrl}" target="_blank" rel="noopener noreferrer"
                onclick="event.preventDefault(); if(event.target.classList.contains('space-install-btn')) return; window.open('${space.installUrl}', '_blank');">
                <div class="space-card-header">
                    <div class="space-likes-bg">
                        <div class="space-likes-label">♥️ ${space.likes}</div>
                    </div>
                </div>

                <div class="space-title">${title}</div>
                <div class="space-description">${description}</div>

                ${this.renderInstallUpdateButton(space, dashboardStatus)}

                <div class="space-meta">
                    <span class="space-by">${space.author}</span>
                    <span class="space-date">${formatDate(space.created)}</span>
                </div>
            </a>
        `;
    }

    renderInstallUpdateButton(space, dashboardStatus) {

        if (Object.keys(dashboardStatus.active_installations).length !== 0) {
            for (const [id, value] of Object.entries(dashboardStatus.active_installations)) {
                if (value.operation === 'install' && value.app_name === space.id) {
                    return `
                        <button class="space-card-btn space-card-installing-btn disabled">
                            Installing
                        </button>
                    `;

                }
                else if (value.operation === 'remove' && value.app_name === space.id) {
                    return `
                        <button class="space-card-btn space-card-removing-btn disabled">
                            Removing
                        </button>
                    `;
                }
            }
        }

        if (dashboardStatus.available_apps.includes(space.id)) {
            return `
            <button class="space-card-btn space-card-installed-btn disabled">
                Installed
            </button>
        `;
        }
        else {
            return `
            <button class="space-card-btn space-card-install-btn" onclick="event.stopPropagation(); installFromSpace('${space.installUrl}', '${space.cardData?.title}'); return false;">
                Install
            </button>
        `;
        }
    }

    async renderSpaces() {
        const dashboardStatus = await (await fetch('/api/status/full')).json();

        const grid = document.getElementById('spaces-grid');

        let spacesCards = this.filteredSpaces.map((space) => {
            return this.renderSpaceCard(space, dashboardStatus);
        }).join('');

        grid.innerHTML = `
            <div class="grid grid-cols-3 md:grid-cols-3 gap-4">
                ${spacesCards}
            </div>
        `;

    }

    showError() {
        const grid = document.getElementById('spaces-grid');
        const stats = document.getElementById('spaces-stats');

        stats.innerHTML = 'Unable to load spaces';
        grid.innerHTML = `
          <div class="col-span-full text-center text-gray-500 py-8">
            <h3 class="font-semibold mb-2">Unable to load Hugging Face Spaces</h3>
            <p class="text-sm">This might be due to CORS restrictions. In a production environment, you'd use a backend API to fetch the data.</p>
          </div>
        `;
    }
}

const formatDate = (timestamp) => {
    const date = new Date(timestamp);

    const now = new Date();
    const diffInDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));

    if (diffInDays === 0) return 'Today';
    if (diffInDays === 1) return '1 day ago';
    if (diffInDays < 30) return `${diffInDays} days ago`;

    let month = date.toLocaleString('default', { month: 'short' });
    let day = date.getDate();
    if (day < 10) day = `0${day}`; // Add leading 0
    let year = date.getFullYear();

    return `${month} ${day}, ${year}`; // e.g., "Jan 01, 2023"
};

initializeSpacesStore(); // Load spaces on page load
setInterval(refreshSpacesStore, 500); // Refresh every minute

