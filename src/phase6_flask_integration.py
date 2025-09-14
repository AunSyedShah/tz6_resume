"""
Phase 6: Flask Integration for Advanced Search & Filter UI
=========================================================

This module integrates the advanced search engine with Flask,
providing API endpoints and web interface enhancements for 
sophisticated search and filtering capabilities.
"""

from flask import Blueprint, request, jsonify, render_template_string
from phase6_advanced_search import AdvancedSearchEngine, SearchFilter, SortCriteria
import json
import logging

logger = logging.getLogger(__name__)

# Create Blueprint for search functionality
search_bp = Blueprint('search', __name__)

# Global search engine instance
search_engine = AdvancedSearchEngine()


@search_bp.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for advanced search and filtering."""
    try:
        data = request.get_json()
        
        # Get candidates from session or load from file
        from ranking_pipeline import RankingPipeline
        pipeline = RankingPipeline()
        candidates = pipeline.load_results()
        
        if not candidates:
            return jsonify({'error': 'No candidates available'}), 400
        
        # Parse search filter
        search_filter = SearchFilter(
            keyword=data.get('keyword', ''),
            score_min=float(data.get('score_min', 0.0)),
            score_max=float(data.get('score_max', 1.0)),
            experience_min=int(data.get('experience_min', 0)),
            experience_max=int(data.get('experience_max', 20)),
            skills_required=data.get('skills_required', []),
            skills_preferred=data.get('skills_preferred', []),
            education_level=data.get('education_level', 'any'),
            certifications=data.get('certifications', []),
            roles=data.get('roles', []),
            domains=data.get('domains', []),
            confidence_min=float(data.get('confidence_min', 0.0)),
            exclude_keywords=data.get('exclude_keywords', [])
        )
        
        # Parse sort criteria
        sort_criteria = None
        if data.get('sort_field'):
            sort_criteria = SortCriteria(
                primary_field=data.get('sort_field', 'final_score'),
                primary_order=data.get('sort_order', 'desc'),
                secondary_field=data.get('sort_field_2'),
                secondary_order=data.get('sort_order_2', 'desc'),
                tertiary_field=data.get('sort_field_3'),
                tertiary_order=data.get('sort_order_3', 'desc')
            )
        
        # Execute search
        filtered_candidates = search_engine.search_candidates(
            candidates, search_filter, sort_criteria
        )
        
        # Prepare response
        response_data = {
            'candidates': filtered_candidates,
            'total_count': len(filtered_candidates),
            'original_count': len(candidates),
            'filter_summary': search_engine._summarize_filter(search_filter)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Search API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/api/search/suggestions', methods=['GET'])
def api_search_suggestions():
    """API endpoint for search suggestions."""
    try:
        query = request.args.get('q', '')
        
        # Get candidates for suggestions
        from ranking_pipeline import RankingPipeline
        pipeline = RankingPipeline()
        candidates = pipeline.load_results()
        
        if not candidates:
            return jsonify({'suggestions': {}})
        
        suggestions = search_engine.get_search_suggestions(candidates, query)
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        logger.error(f"Suggestions API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/api/search/presets', methods=['GET'])
def api_get_presets():
    """API endpoint to get available filter presets."""
    try:
        presets = search_engine.get_available_presets()
        return jsonify({'presets': presets})
        
    except Exception as e:
        logger.error(f"Presets API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/api/search/presets', methods=['POST'])
def api_save_preset():
    """API endpoint to save a filter preset."""
    try:
        data = request.get_json()
        preset_name = data.get('name')
        filter_config = data.get('filter')
        
        if not preset_name or not filter_config:
            return jsonify({'error': 'Name and filter configuration required'}), 400
        
        # Create SearchFilter from config
        search_filter = SearchFilter(**filter_config)
        search_engine.save_filter_preset(preset_name, search_filter)
        
        return jsonify({'success': True, 'message': f'Preset "{preset_name}" saved'})
        
    except Exception as e:
        logger.error(f"Save preset API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/api/search/presets/<preset_name>', methods=['GET'])
def api_get_preset(preset_name):
    """API endpoint to get a specific filter preset."""
    try:
        preset = search_engine.get_filter_preset(preset_name)
        
        if not preset:
            return jsonify({'error': 'Preset not found'}), 404
        
        return jsonify({'preset': preset.__dict__})
        
    except Exception as e:
        logger.error(f"Get preset API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/api/search/history', methods=['GET'])
def api_search_history():
    """API endpoint to get search history."""
    try:
        return jsonify({'history': search_engine.search_history})
        
    except Exception as e:
        logger.error(f"Search history API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Enhanced search interface JavaScript
ADVANCED_SEARCH_JS = """
class AdvancedSearchManager {
    constructor() {
        this.currentFilter = {};
        this.searchTimeout = null;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadPresets();
        this.enableRealTimeSearch();
    }
    
    setupEventListeners() {
        // Main search input
        const searchInput = document.getElementById('advancedSearchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.handleSearchInput(e.target.value);
            });
        }
        
        // Filter controls
        document.querySelectorAll('.filter-control').forEach(control => {
            control.addEventListener('change', () => {
                this.updateFilters();
                this.executeSearch();
            });
        });
        
        // Preset selection
        const presetSelect = document.getElementById('presetSelect');
        if (presetSelect) {
            presetSelect.addEventListener('change', (e) => {
                this.loadPreset(e.target.value);
            });
        }
        
        // Save preset button
        const savePresetBtn = document.getElementById('savePresetBtn');
        if (savePresetBtn) {
            savePresetBtn.addEventListener('click', () => {
                this.showSavePresetModal();
            });
        }
        
        // Clear filters button
        const clearFiltersBtn = document.getElementById('clearFiltersBtn');
        if (clearFiltersBtn) {
            clearFiltersBtn.addEventListener('click', () => {
                this.clearAllFilters();
            });
        }
    }
    
    handleSearchInput(query) {
        // Clear previous timeout
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
        }
        
        // Debounce search
        this.searchTimeout = setTimeout(() => {
            this.updateFilters();
            this.executeSearch();
            this.loadSuggestions(query);
        }, 300);
    }
    
    loadSuggestions(query) {
        if (query.length < 2) return;
        
        fetch(`/api/search/suggestions?q=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                this.displaySuggestions(data.suggestions);
            })
            .catch(error => {
                console.error('Error loading suggestions:', error);
            });
    }
    
    displaySuggestions(suggestions) {
        const suggestionsList = document.getElementById('searchSuggestions');
        if (!suggestionsList) return;
        
        suggestionsList.innerHTML = '';
        
        Object.entries(suggestions).forEach(([category, items]) => {
            if (items.length > 0) {
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'suggestion-category';
                categoryDiv.innerHTML = `
                    <strong>${category.charAt(0).toUpperCase() + category.slice(1)}:</strong>
                    ${items.map(item => `<span class="suggestion-item" onclick="advancedSearch.selectSuggestion('${item}')">${item}</span>`).join(', ')}
                `;
                suggestionsList.appendChild(categoryDiv);
            }
        });
        
        suggestionsList.style.display = suggestions && Object.keys(suggestions).length > 0 ? 'block' : 'none';
    }
    
    selectSuggestion(suggestion) {
        const searchInput = document.getElementById('advancedSearchInput');
        if (searchInput) {
            searchInput.value = suggestion;
            this.handleSearchInput(suggestion);
        }
        document.getElementById('searchSuggestions').style.display = 'none';
    }
    
    updateFilters() {
        this.currentFilter = {
            keyword: document.getElementById('advancedSearchInput')?.value || '',
            score_min: parseFloat(document.getElementById('scoreMin')?.value || 0),
            score_max: parseFloat(document.getElementById('scoreMax')?.value || 1),
            experience_min: parseInt(document.getElementById('experienceMin')?.value || 0),
            experience_max: parseInt(document.getElementById('experienceMax')?.value || 20),
            skills_required: this.getMultiSelectValues('skillsRequired'),
            skills_preferred: this.getMultiSelectValues('skillsPreferred'),
            education_level: document.getElementById('educationLevel')?.value || 'any',
            certifications: this.getMultiSelectValues('certifications'),
            roles: this.getMultiSelectValues('roles'),
            domains: this.getMultiSelectValues('domains'),
            confidence_min: parseFloat(document.getElementById('confidenceMin')?.value || 0),
            exclude_keywords: this.getArrayFromInput('excludeKeywords'),
            sort_field: document.getElementById('sortField')?.value || 'final_score',
            sort_order: document.getElementById('sortOrder')?.value || 'desc',
            sort_field_2: document.getElementById('sortField2')?.value || '',
            sort_order_2: document.getElementById('sortOrder2')?.value || 'desc'
        };
        
        this.updateFilterSummary();
    }
    
    getMultiSelectValues(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return [];
        
        if (element.multiple) {
            return Array.from(element.selectedOptions).map(option => option.value);
        } else {
            return element.value ? [element.value] : [];
        }
    }
    
    getArrayFromInput(elementId) {
        const element = document.getElementById(elementId);
        if (!element || !element.value) return [];
        
        return element.value.split(',').map(item => item.trim()).filter(item => item);
    }
    
    executeSearch() {
        const loadingIndicator = document.getElementById('searchLoading');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }
        
        fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(this.currentFilter)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            this.displayResults(data);
        })
        .catch(error => {
            console.error('Search error:', error);
            this.showError('Search failed: ' + error.message);
        })
        .finally(() => {
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
        });
    }
    
    displayResults(data) {
        const resultsContainer = document.getElementById('searchResults');
        if (!resultsContainer) return;
        
        // Update result count
        const resultCount = document.getElementById('resultCount');
        if (resultCount) {
            resultCount.textContent = `${data.total_count} of ${data.original_count} candidates`;
        }
        
        // Display filter summary
        const filterSummary = document.getElementById('filterSummary');
        if (filterSummary) {
            filterSummary.textContent = data.filter_summary;
        }
        
        // Display candidates
        if (data.candidates.length === 0) {
            resultsContainer.innerHTML = '<div class="alert alert-info">No candidates match the current filters.</div>';
            return;
        }
        
        const candidatesHtml = data.candidates.map(candidate => this.renderCandidate(candidate)).join('');
        resultsContainer.innerHTML = candidatesHtml;
        
        // Update export data
        window.currentCandidates = data.candidates;
    }
    
    renderCandidate(candidate) {
        const score = (candidate.final_score || candidate.Score || 0).toFixed(3);
        const confidence = (candidate.confidence || 0.8).toFixed(3);
        const skills = (candidate.All_Skills || []).slice(0, 5).join(', ');
        const roles = (candidate.All_Roles || []).slice(0, 3).join(', ');
        
        return `
            <div class="candidate-card border rounded p-3 mb-3">
                <div class="row">
                    <div class="col-md-8">
                        <h5 class="candidate-name">${candidate.filename}</h5>
                        <p class="candidate-roles text-muted">${roles}</p>
                        <p class="candidate-skills"><strong>Skills:</strong> ${skills}</p>
                        ${candidate.preferred_skill_matches ? `<span class="badge badge-success">+${candidate.preferred_skill_matches} preferred skills</span>` : ''}
                    </div>
                    <div class="col-md-4 text-right">
                        <div class="candidate-score">
                            <strong>Score: ${score}</strong>
                            <br>
                            <small>Confidence: ${confidence}</small>
                        </div>
                        <div class="candidate-actions mt-2">
                            <button class="btn btn-sm btn-outline-primary" onclick="viewCandidate('${candidate.filename}')">
                                View Details
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    loadPresets() {
        fetch('/api/search/presets')
            .then(response => response.json())
            .then(data => {
                const presetSelect = document.getElementById('presetSelect');
                if (presetSelect) {
                    presetSelect.innerHTML = '<option value="">Select a preset...</option>';
                    Object.entries(data.presets).forEach(([name, description]) => {
                        const option = document.createElement('option');
                        option.value = name;
                        option.textContent = `${name} - ${description}`;
                        presetSelect.appendChild(option);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading presets:', error);
            });
    }
    
    loadPreset(presetName) {
        if (!presetName) return;
        
        fetch(`/api/search/presets/${presetName}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                this.applyFilterPreset(data.preset);
            })
            .catch(error => {
                console.error('Error loading preset:', error);
                this.showError('Failed to load preset: ' + error.message);
            });
    }
    
    applyFilterPreset(preset) {
        // Apply preset values to form controls
        if (document.getElementById('advancedSearchInput')) {
            document.getElementById('advancedSearchInput').value = preset.keyword || '';
        }
        if (document.getElementById('scoreMin')) {
            document.getElementById('scoreMin').value = preset.score_min || 0;
        }
        if (document.getElementById('scoreMax')) {
            document.getElementById('scoreMax').value = preset.score_max || 1;
        }
        if (document.getElementById('experienceMin')) {
            document.getElementById('experienceMin').value = preset.experience_min || 0;
        }
        if (document.getElementById('experienceMax')) {
            document.getElementById('experienceMax').value = preset.experience_max || 20;
        }
        if (document.getElementById('confidenceMin')) {
            document.getElementById('confidenceMin').value = preset.confidence_min || 0;
        }
        if (document.getElementById('educationLevel')) {
            document.getElementById('educationLevel').value = preset.education_level || 'any';
        }
        
        // Update filters and execute search
        this.updateFilters();
        this.executeSearch();
    }
    
    showSavePresetModal() {
        const modal = document.getElementById('savePresetModal');
        if (modal) {
            $(modal).modal('show');
        }
    }
    
    savePreset() {
        const nameInput = document.getElementById('presetName');
        const name = nameInput?.value.trim();
        
        if (!name) {
            alert('Please enter a preset name');
            return;
        }
        
        fetch('/api/search/presets', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                filter: this.currentFilter
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            alert('Preset saved successfully!');
            this.loadPresets();
            $('#savePresetModal').modal('hide');
            nameInput.value = '';
        })
        .catch(error => {
            console.error('Error saving preset:', error);
            alert('Failed to save preset: ' + error.message);
        });
    }
    
    clearAllFilters() {
        // Reset all form controls
        document.querySelectorAll('.filter-control').forEach(control => {
            if (control.type === 'text' || control.type === 'number') {
                control.value = control.defaultValue || '';
            } else if (control.type === 'select-one') {
                control.selectedIndex = 0;
            } else if (control.type === 'select-multiple') {
                control.selectedIndex = -1;
            }
        });
        
        // Clear search input
        const searchInput = document.getElementById('advancedSearchInput');
        if (searchInput) {
            searchInput.value = '';
        }
        
        // Update and execute search
        this.updateFilters();
        this.executeSearch();
    }
    
    updateFilterSummary() {
        const summary = document.getElementById('activeFiltersSummary');
        if (!summary) return;
        
        const activeFilters = [];
        
        if (this.currentFilter.keyword) {
            activeFilters.push(`Keyword: "${this.currentFilter.keyword}"`);
        }
        if (this.currentFilter.score_min > 0 || this.currentFilter.score_max < 1) {
            activeFilters.push(`Score: ${this.currentFilter.score_min}-${this.currentFilter.score_max}`);
        }
        if (this.currentFilter.experience_min > 0 || this.currentFilter.experience_max < 20) {
            activeFilters.push(`Experience: ${this.currentFilter.experience_min}-${this.currentFilter.experience_max} years`);
        }
        if (this.currentFilter.skills_required.length > 0) {
            activeFilters.push(`Required skills: ${this.currentFilter.skills_required.join(', ')}`);
        }
        
        summary.innerHTML = activeFilters.length > 0 
            ? `<strong>Active filters:</strong> ${activeFilters.join(' | ')}` 
            : '<em>No active filters</em>';
    }
    
    enableRealTimeSearch() {
        // Enable real-time updates for range sliders
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            slider.addEventListener('input', () => {
                this.updateFilters();
                this.executeSearch();
            });
        });
    }
    
    showError(message) {
        const errorContainer = document.getElementById('searchErrors');
        if (errorContainer) {
            errorContainer.innerHTML = `<div class="alert alert-danger">${message}</div>`;
            setTimeout(() => {
                errorContainer.innerHTML = '';
            }, 5000);
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    window.advancedSearch = new AdvancedSearchManager();
});
"""


def get_advanced_search_css():
    """Get CSS styles for advanced search interface."""
    return """
    .advanced-search-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .filter-section {
        margin-bottom: 20px;
        padding: 15px;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        background: white;
    }
    
    .filter-section h6 {
        color: #495057;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid #007bff;
        padding-bottom: 5px;
    }
    
    .search-suggestions {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        z-index: 1000;
        max-height: 200px;
        overflow-y: auto;
        display: none;
    }
    
    .suggestion-category {
        padding: 8px 12px;
        border-bottom: 1px solid #eee;
    }
    
    .suggestion-item {
        background: #e9ecef;
        padding: 2px 6px;
        margin: 2px;
        border-radius: 3px;
        cursor: pointer;
        display: inline-block;
    }
    
    .suggestion-item:hover {
        background: #007bff;
        color: white;
    }
    
    .candidate-card {
        transition: all 0.2s ease;
        background: white;
    }
    
    .candidate-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .candidate-score {
        font-size: 1.1em;
    }
    
    .filter-control {
        margin-bottom: 10px;
    }
    
    .range-value {
        font-weight: 600;
        color: #007bff;
    }
    
    .search-stats {
        background: #e9ecef;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    
    .active-filters {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    
    .preset-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .search-loading {
        text-align: center;
        padding: 20px;
        color: #6c757d;
    }
    
    .multi-select {
        min-height: 100px;
    }
    
    @media (max-width: 768px) {
        .advanced-search-container {
            padding: 15px;
        }
        
        .filter-section {
            padding: 10px;
        }
        
        .candidate-card .row {
            flex-direction: column;
        }
        
        .candidate-card .col-md-4 {
            margin-top: 10px;
            text-align: left !important;
        }
    }
    """