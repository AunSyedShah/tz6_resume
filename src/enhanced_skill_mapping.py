"""
Enhanced Skill Mapping System
Phase 1: Comprehensive skill synonyms, abbreviations, and variations
"""

class SkillSynonymMapper:
    def __init__(self):
        self.skill_mappings = {
            # Programming Languages
            'javascript': ['js', 'ecmascript', 'node.js', 'nodejs', 'react.js', 'reactjs'],
            'python': ['py', 'python3', 'django', 'flask', 'fastapi'],
            'java': ['j2ee', 'jee', 'java ee', 'spring boot', 'springboot'],
            'csharp': ['c#', '.net', 'dotnet', 'asp.net', 'aspnet'],
            'cplusplus': ['c++', 'cpp'],
            'typescript': ['ts'],
            'php': ['php7', 'php8', 'laravel', 'symfony'],
            'ruby': ['rails', 'ruby on rails', 'ror'],
            'golang': ['go', 'go lang'],
            'rust': ['rust lang'],
            'kotlin': ['kotlin jvm'],
            'swift': ['swift ui', 'swiftui'],
            'scala': ['scala lang'],
            
            # Frontend Technologies
            'react': ['reactjs', 'react.js', 'react native', 'reactnative'],
            'angular': ['angularjs', 'angular.js', 'angular 2+'],
            'vue': ['vuejs', 'vue.js', 'nuxt', 'nuxtjs'],
            'html': ['html5', 'html 5', 'markup'],
            'css': ['css3', 'css 3', 'scss', 'sass', 'less'],
            'bootstrap': ['bootstrap 4', 'bootstrap 5', 'bs4', 'bs5'],
            'jquery': ['jquery ui', 'jqueryui'],
            
            # Backend Technologies
            'spring': ['spring boot', 'spring mvc', 'spring framework'],
            'express': ['expressjs', 'express.js'],
            'django': ['django rest', 'drf'],
            'flask': ['flask api'],
            'fastapi': ['fast api'],
            'nodejs': ['node.js', 'node js'],
            
            # Databases
            'mysql': ['my sql', 'mariadb'],
            'postgresql': ['postgres', 'psql'],
            'mongodb': ['mongo db', 'mongo', 'nosql'],
            'redis': ['redis cache'],
            'elasticsearch': ['elastic search', 'elk'],
            'cassandra': ['apache cassandra'],
            'oracle': ['oracle db', 'oracle database', 'plsql', 'pl/sql'],
            'sql server': ['mssql', 'ms sql', 'sqlserver', 't-sql', 'tsql'],
            'sqlite': ['sqlite3'],
            
            # Cloud Platforms
            'aws': ['amazon web services', 'amazon aws', 'ec2', 's3', 'lambda', 'rds'],
            'azure': ['microsoft azure', 'azure cloud'],
            'gcp': ['google cloud', 'google cloud platform', 'gce'],
            'kubernetes': ['k8s', 'kube'],
            'docker': ['containerization', 'containers'],
            
            # DevOps & Tools
            'jenkins': ['ci/cd', 'continuous integration'],
            'git': ['github', 'gitlab', 'bitbucket', 'version control'],
            'terraform': ['infrastructure as code', 'iac'],
            'ansible': ['configuration management'],
            'maven': ['build tool'],
            'gradle': ['build automation'],
            'npm': ['node package manager'],
            'webpack': ['module bundler'],
            
            # AI/ML Technologies
            'machine learning': ['ml', 'artificial intelligence', 'ai', 'deep learning', 'dl'],
            'tensorflow': ['tf', 'keras'],
            'pytorch': ['torch'],
            'scikit-learn': ['sklearn', 'scikit learn'],
            'pandas': ['data analysis'],
            'numpy': ['numerical computing'],
            'jupyter': ['jupyter notebook', 'ipython'],
            
            # Testing Frameworks
            'selenium': ['automation testing', 'web testing'],
            'junit': ['java testing', 'unit testing'],
            'pytest': ['python testing'],
            'jest': ['javascript testing'],
            'cypress': ['e2e testing'],
            
            # Business Analysis
            'business analysis': ['ba', 'business analyst', 'requirements analysis'],
            'agile': ['scrum', 'kanban', 'sprint'],
            'waterfall': ['sdlc', 'traditional methodology'],
            'jira': ['issue tracking', 'project management'],
            'confluence': ['documentation'],
            'visio': ['process modeling', 'diagramming'],
            
            # Healthcare Domain
            'hipaa': ['health insurance portability', 'healthcare compliance'],
            'hl7': ['health level 7', 'healthcare standards'],
            'emr': ['electronic medical records', 'ehr', 'electronic health records'],
            'icd': ['icd-9', 'icd-10', 'medical coding'],
            'facets': ['healthcare claims processing'],
            
            # Finance Domain
            'fintech': ['financial technology'],
            'blockchain': ['cryptocurrency', 'bitcoin', 'ethereum'],
            'trading': ['algorithmic trading', 'quantitative analysis'],
            'risk management': ['credit risk', 'market risk'],
            
            # Certifications
            'pmp': ['project management professional'],
            'aws certified': ['aws certification', 'cloud certification'],
            'oracle certified': ['oca', 'ocp'],
            'microsoft certified': ['mcsa', 'mcse'],
            'cissp': ['information security'],
            
            # Soft Skills
            'leadership': ['team lead', 'team leader', 'management'],
            'communication': ['stakeholder management', 'client interaction'],
            'problem solving': ['analytical thinking', 'troubleshooting'],
        }
        
        # Create reverse mapping for fast lookup
        self.synonym_to_canonical = {}
        for canonical, synonyms in self.skill_mappings.items():
            self.synonym_to_canonical[canonical] = canonical
            for synonym in synonyms:
                self.synonym_to_canonical[synonym.lower()] = canonical
    
    def normalize_skill(self, skill):
        """Convert skill to its canonical form"""
        skill_lower = skill.lower().strip()
        return self.synonym_to_canonical.get(skill_lower, skill_lower)
    
    def get_all_variations(self, skill):
        """Get all variations of a skill including the canonical form"""
        canonical = self.normalize_skill(skill)
        variations = [canonical]
        if canonical in self.skill_mappings:
            variations.extend(self.skill_mappings[canonical])
        return list(set(variations))
    
    def expand_skills_list(self, skills):
        """Expand a list of skills to include all variations"""
        expanded = set()
        for skill in skills:
            variations = self.get_all_variations(skill)
            expanded.update(variations)
        return list(expanded)

class EnhancedSkillExtractor:
    def __init__(self):
        self.mapper = SkillSynonymMapper()
        
        # Common skill patterns - refined to avoid over-matching
        self.skill_patterns = [
            # Years of experience patterns
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience\s+)?(?:in\s+|with\s+)([a-zA-Z0-9\s\-\.#\+]{2,30})(?:\s|,|\.|\n|$)',
            
            # Technology stack patterns
            r'expertise\s+in\s+([a-zA-Z0-9\s\-\.#\+]{2,30})(?:\s|,|\.|\n|$)',
            r'proficient\s+(?:in\s+|with\s+)([a-zA-Z0-9\s\-\.#\+]{2,30})(?:\s|,|\.|\n|$)',
            r'experienced\s+(?:in\s+|with\s+)([a-zA-Z0-9\s\-\.#\+]{2,30})(?:\s|,|\.|\n|$)',
            r'knowledge\s+of\s+([a-zA-Z0-9\s\-\.#\+]{2,30})(?:\s|,|\.|\n|$)',
            r'skilled\s+in\s+([a-zA-Z0-9\s\-\.#\+]{2,30})(?:\s|,|\.|\n|$)',
            r'hands[\-\s]on\s+experience\s+(?:with\s+|in\s+)([a-zA-Z0-9\s\-\.#\+]{2,30})(?:\s|,|\.|\n|$)',
            
            # Framework/Tool patterns
            r'using\s+([a-zA-Z0-9\s\-\.#\+]{2,20})(?:\s|,|\.|\n|$)',
            r'worked\s+with\s+([a-zA-Z0-9\s\-\.#\+]{2,20})(?:\s|,|\.|\n|$)',
        ]
    
    def extract_enhanced_skills(self, text):
        """Extract skills with enhanced pattern matching and normalization"""
        import re
        
        skills = set()
        text_lower = text.lower()
        
        # Extract using patterns
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Get the skill part (last capturing group that exists)
                groups = match.groups()
                if groups:
                    skill_text = groups[-1].strip()
                    
                    # Split by common delimiters
                    skill_parts = re.split(r'[,;/\|\&\n\r]+', skill_text)
                    
                    for part in skill_parts:
                        part = part.strip(' \t\n\r.,;')
                        # Filter out noise: too short, only digits, common words
                        if (len(part) > 2 and not part.isdigit() and 
                            part not in ['and', 'or', 'the', 'with', 'using', 'in', 'of', 'for']):
                            # Normalize the skill
                            normalized = self.mapper.normalize_skill(part)
                            if len(normalized) > 2:  # Ensure normalized skill is meaningful
                                skills.add(normalized)
        
        # Common technology keywords (fallback)
        common_tech = [
            'python', 'java', 'javascript', 'react', 'angular', 'node.js', 'sql',
            'aws', 'docker', 'kubernetes', 'git', 'jenkins', 'agile', 'scrum',
            'machine learning', 'tensorflow', 'pytorch', 'mongodb', 'postgresql',
            'spring boot', 'django', 'flask', 'html', 'css', 'bootstrap'
        ]
        
        for tech in common_tech:
            if tech in text_lower:
                normalized = self.mapper.normalize_skill(tech)
                skills.add(normalized)
        
        return list(skills)

# Test the enhanced skill extraction
if __name__ == "__main__":
    extractor = EnhancedSkillExtractor()
    
    sample_text = """
    Senior Java Developer with 5+ years of experience in Spring Boot, React.js, and AWS.
    Proficient in JavaScript, Node.js, and MongoDB. Expertise in Docker, Kubernetes, and CI/CD.
    Hands-on experience with machine learning using Python and TensorFlow.
    """
    
    skills = extractor.extract_enhanced_skills(sample_text)
    print("Enhanced Skills Extracted:")
    for skill in sorted(skills):
        print(f"  - {skill}")