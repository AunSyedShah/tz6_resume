// AI Resume Ranker Pro - JavaScript Functions

// Toggle JD input method
function toggleJDInput() {
    const textInput = document.getElementById('jd_text_input');
    const fileInput = document.getElementById('jd_file_input');
    const textRadio = document.getElementById('jd_text');
    
    if (textRadio.checked) {
        textInput.style.display = 'block';
        fileInput.style.display = 'none';
    } else {
        textInput.style.display = 'none';
        fileInput.style.display = 'block';
    }
}

// Toggle advanced matching options
function toggleAdvancedOptions() {
    const advancedOptions = document.getElementById('advanced_options');
    const checkbox = document.getElementById('use_advanced_matching');
    
    if (checkbox.checked) {
        advancedOptions.style.display = 'block';
    } else {
        advancedOptions.style.display = 'none';
    }
}

// Update threshold value display
function updateThresholdValue(value) {
    document.getElementById('threshold-value').textContent = value;
}

// Update experience min value display
function updateExpMinValue(value) {
    document.getElementById('exp-min-value').textContent = value + ' years';
}

// Update experience max value display
function updateExpMaxValue(value) {
    document.getElementById('exp-max-value').textContent = value + ' years';
}

// Update rating value display
function updateRatingValue(value) {
    const ratingLabels = ['', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent'];
    document.getElementById('rating-value').textContent = value + ' - ' + ratingLabels[value];
}

// Load filter options from server
function loadFilterOptions() {
    fetch('/get_filter_options')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading filter options:', data.error);
                return;
            }
            
            // Update skills dropdown
            const skillSelect = document.getElementById('filter_skill');
            skillSelect.innerHTML = '<option value="All">All Skills</option>';
            data.skills.forEach(skill => {
                const option = document.createElement('option');
                option.value = skill;
                option.textContent = skill;
                skillSelect.appendChild(option);
            });
            
            // Update roles dropdown
            const roleSelect = document.getElementById('filter_role');
            roleSelect.innerHTML = '<option value="All">All Roles</option>';
            data.roles.forEach(role => {
                const option = document.createElement('option');
                option.value = role;
                option.textContent = role;
                roleSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error loading filter options:', error);
        });
}

// Load and display results table
function loadResultsTable() {
    const resultsContainer = document.getElementById('results-table');
    if (!resultsContainer) return;
    
    // This would typically make an AJAX call to get filtered results
    // For now, we'll implement basic table structure
    const results = window.sessionResults || [];
    
    if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="text-muted text-center">No results to display.</p>';
        return;
    }
    
    let tableHTML = `
        <div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Filename</th>
                        <th>Score</th>
                        <th>Method</th>
                        <th>Skills</th>
                        <th>Roles</th>
                        <th>Education</th>
                        <th>Experience</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    results.forEach((result, index) => {
        const scoreClass = result.Score >= 80 ? 'score-high' : 
                          result.Score >= 60 ? 'score-medium' : 'score-low';
        
        tableHTML += `
            <tr>
                <td><span class="badge bg-primary">${index + 1}</span></td>
                <td><strong>${result.Filename}</strong></td>
                <td class="${scoreClass}">${result.Score}%</td>
                <td><small class="text-muted">${result.Method}</small></td>
                <td><small>${result.Skills}</small></td>
                <td><small>${result.Roles}</small></td>
                <td><small>${result.Education}</small></td>
                <td>${result.Experience} years</td>
            </tr>
        `;
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    resultsContainer.innerHTML = tableHTML;
}

// Show loading state for forms
function showLoading(formElement) {
    const submitButton = formElement.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
    formElement.classList.add('loading');
}

// Hide loading state for forms
function hideLoading(formElement) {
    const submitButton = formElement.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = false;
        // Restore original button text based on context
        if (submitButton.closest('form').action.includes('evaluate')) {
            submitButton.innerHTML = '<i class="fas fa-rocket"></i> 🚀 Evaluate Resumes';
        } else if (submitButton.closest('form').action.includes('upload_jd')) {
            submitButton.innerHTML = '<i class="fas fa-upload"></i> Load JD';
        } else if (submitButton.closest('form').action.includes('upload_resumes')) {
            submitButton.innerHTML = '<i class="fas fa-upload"></i> Upload Resumes';
        } else {
            submitButton.innerHTML = submitButton.innerHTML.replace('<i class="fas fa-spinner fa-spin"></i> Processing...', submitButton.textContent);
        }
    }
    formElement.classList.remove('loading');
}

// Handle form submissions with loading states
function handleFormSubmission(event) {
    const form = event.target;
    showLoading(form);
    
    // The form will submit normally, and the page will reload
    // Loading state will be reset on page reload
}

// Initialize page functionality
document.addEventListener('DOMContentLoaded', function() {
    // Load filter options if results exist
    if (document.getElementById('results-container')) {
        loadFilterOptions();
        loadResultsTable();
    }
    
    // Add loading state to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', handleFormSubmission);
    });
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.classList.contains('show')) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    });
});

// File upload preview
function previewFiles(input, previewContainer) {
    const files = input.files;
    const container = document.getElementById(previewContainer);
    
    if (!container) return;
    
    container.innerHTML = '';
    
    Array.from(files).forEach(file => {
        const fileDiv = document.createElement('div');
        fileDiv.className = 'mb-2 p-2 border rounded';
        fileDiv.innerHTML = `
            <i class="fas fa-file-word text-primary"></i>
            <span class="ms-2">${file.name}</span>
            <small class="text-muted ms-2">(${(file.size / 1024 / 1024).toFixed(2)} MB)</small>
        `;
        container.appendChild(fileDiv);
    });
}

// Progress bar animation
function animateProgressBar(element, targetValue) {
    let currentValue = 0;
    const increment = targetValue / 50; // Animate over 50 steps
    
    const timer = setInterval(() => {
        currentValue += increment;
        if (currentValue >= targetValue) {
            currentValue = targetValue;
            clearInterval(timer);
        }
        element.style.width = currentValue + '%';
        element.setAttribute('aria-valuenow', currentValue);
    }, 20);
}

// Smooth scroll to element
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Export functionality
function exportResults(format) {
    const form = document.createElement('form');
    form.method = 'GET';
    form.action = `/download/${format}`;
    document.body.appendChild(form);
    form.submit();
    document.body.removeChild(form);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+E or Cmd+E to evaluate resumes
    if ((event.ctrlKey || event.metaKey) && event.key === 'e') {
        event.preventDefault();
        const evaluateButton = document.querySelector('button[type="submit"]');
        if (evaluateButton && evaluateButton.textContent.includes('Evaluate')) {
            evaluateButton.click();
        }
    }
    
    // Ctrl+U or Cmd+U to focus on file upload
    if ((event.ctrlKey || event.metaKey) && event.key === 'u') {
        event.preventDefault();
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.focus();
        }
    }
});