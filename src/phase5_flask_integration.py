"""
Flask Integration for Phase 5: Export & Reporting System
=======================================================

This module integrates the export functionality into the Flask application
with API endpoints for generating Excel and PDF reports.
"""

from flask import Blueprint, request, jsonify, send_file, current_app
import os
import json
from datetime import datetime
import tempfile
import logging

# Import our export system
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase5_export_system import CandidateExporter

logger = logging.getLogger(__name__)

# Create Blueprint for export routes
export_bp = Blueprint('export', __name__)


@export_bp.route('/export/excel', methods=['POST'])
def export_excel():
    """Export candidate rankings to Excel format."""
    try:
        data = request.get_json()
        
        if not data or 'candidates' not in data or 'job_description' not in data:
            return jsonify({'error': 'Missing required data: candidates and job_description'}), 400
        
        candidates = data['candidates']
        job_description = data['job_description']
        filename = data.get('filename')
        
        # Initialize exporter
        exporter = CandidateExporter(output_dir=current_app.config.get('EXPORT_DIR', 'exports'))
        
        # Generate Excel file
        filepath = exporter.export_to_excel(candidates, job_description, filename)
        
        # Return file for download
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"Excel export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@export_bp.route('/export/pdf', methods=['POST'])
def export_pdf():
    """Export candidate rankings to PDF format."""
    try:
        data = request.get_json()
        
        if not data or 'candidates' not in data or 'job_description' not in data:
            return jsonify({'error': 'Missing required data: candidates and job_description'}), 400
        
        candidates = data['candidates']
        job_description = data['job_description']
        filename = data.get('filename')
        
        # Initialize exporter
        exporter = CandidateExporter(output_dir=current_app.config.get('EXPORT_DIR', 'exports'))
        
        # Generate PDF file
        filepath = exporter.export_to_pdf(candidates, job_description, filename)
        
        # Return file for download
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@export_bp.route('/export/summary', methods=['POST'])
def get_summary():
    """Get summary statistics for candidates."""
    try:
        data = request.get_json()
        
        if not data or 'candidates' not in data or 'job_description' not in data:
            return jsonify({'error': 'Missing required data: candidates and job_description'}), 400
        
        candidates = data['candidates']
        job_description = data['job_description']
        
        # Initialize exporter
        exporter = CandidateExporter()
        
        # Generate summary
        summary = exporter.create_summary_report(candidates, job_description)
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        return jsonify({'error': f'Summary failed: {str(e)}'}), 500


@export_bp.route('/export/formats', methods=['GET'])
def get_export_formats():
    """Get available export formats and their descriptions."""
    formats = {
        'excel': {
            'name': 'Excel Spreadsheet',
            'description': 'Comprehensive candidate rankings with multiple sheets including analytics',
            'features': [
                'Candidate rankings with detailed scores',
                'Summary analytics and statistics',
                'Detailed feature analysis',
                'Job description analysis',
                'Professional formatting with charts'
            ],
            'file_extension': '.xlsx',
            'endpoint': '/export/excel'
        },
        'pdf': {
            'name': 'PDF Report',
            'description': 'Professional report suitable for presentations and stakeholder review',
            'features': [
                'Executive summary',
                'Top 10 candidates table',
                'Analytics and insights',
                'Professional formatting',
                'Ready for presentation'
            ],
            'file_extension': '.pdf',
            'endpoint': '/export/pdf'
        },
        'summary': {
            'name': 'JSON Summary',
            'description': 'Statistical summary in JSON format for further processing',
            'features': [
                'Key statistics',
                'Score distributions',
                'Recommendation counts',
                'Processing metadata'
            ],
            'file_extension': '.json',
            'endpoint': '/export/summary'
        }
    }
    
    return jsonify(formats)


# Utility functions for the main Flask app
def prepare_candidates_for_export(ranked_resumes, matching_results=None):
    """Prepare candidate data for export."""
    candidates = []
    
    for idx, resume in enumerate(ranked_resumes):
        candidate_data = {
            'filename': resume.get('filename', f'candidate_{idx+1}'),
            'final_score': resume.get('score', 0),
            'confidence': resume.get('confidence', 0.8),  # Default confidence
            'features': {
                'skill_match': resume.get('skill_match', 0),
                'experience_match': resume.get('experience_match', 0),
                'seniority_match': resume.get('seniority_match', 0),
                'domain_relevance': resume.get('domain_relevance', 0),
                'technical_depth': resume.get('technical_depth', 0),
                'certification_match': resume.get('certification_match', 0),
                'education_match': resume.get('education_match', 0)
            },
            'algorithm_scores': resume.get('algorithm_scores', {})
        }
        candidates.append(candidate_data)
    
    return candidates


def add_export_context(context_dict):
    """Add export-related context to template rendering."""
    context_dict.update({
        'export_available': True,
        'export_formats': ['excel', 'pdf', 'summary'],
        'export_descriptions': {
            'excel': 'Detailed spreadsheet with analytics',
            'pdf': 'Professional presentation report',
            'summary': 'Statistical summary data'
        }
    })
    return context_dict