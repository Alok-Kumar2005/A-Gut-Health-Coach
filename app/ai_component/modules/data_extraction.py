import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
import os
from typing import List, Dict, Optional

class GutHealthDataExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.data_sources = {
            'healthline': [
                "https://www.healthline.com/nutrition/gut-microbiome-and-health",
                "https://www.healthline.com/health/gut-health",
                "https://www.healthline.com/health/digestive-health/3-day-gut-reset",
                "https://www.healthline.com/health/10-gut-foods",
                "https://www.healthline.com/nutrition/microbiome-diet",
                "https://www.healthline.com/nutrition/9-signs-and-symptoms-of-ibs",
                "https://www.healthline.com/health/irritable-bowel-syndrome",
                "https://www.healthline.com/health/irritable-bowel-syndrome/ibs-d-diagnosis-treatment-options",
                "https://www.healthline.com/health/ibs-constipation",
                "https://www.healthline.com/health/irritable-bowel-syndrome/ibs-bloating-treatments",
                "https://www.healthline.com/health/digestive-health/foods-to-avoid-with-ibs",
                "https://www.healthline.com/health/ibs/ibs-flare-up-treatment",
                "https://www.healthline.com/nutrition/probiotics-101",
                "https://www.healthline.com/health/digestive-health/dysbiosis",
                "https://www.healthline.com/health/microbiome-testing",
                "https://www.healthline.com/nutrition/19-best-prebiotic-foods",
                "https://www.healthline.com/health/digestive-health/best-probiotic-foods"
            ],
            'mayo_clinic': [
                "https://www.mayoclinic.org/diseases-conditions/irritable-bowel-syndrome/symptoms-causes/syc-20360016",
                "https://www.mayoclinic.org/diseases-conditions/irritable-bowel-syndrome/diagnosis-treatment/drc-20360064",
                "https://connect.mayoclinic.org/blog/weight-management-1/newsfeed-post/building-a-healthy-gut-microbiome/",
                "https://communityhealth.mayoclinic.org/featured-stories/healthy-gut-microbiome",
                "https://mcpress.mayoclinic.org/dairy-health/prebiotics-probiotics-and-the-microbes-in-your-gut-key-to-your-digestive-health/",
                "https://www.mayoclinic.org/medical-professionals/digestive-diseases/news/the-microbiome-fecal-microbiota-transplants-and-inflammatory-bowel-disease/mqc-20463208",
                "https://newsnetwork.mayoclinic.org/discussion/mayo-researcher-finds-potential-microbial-pathway-to-treat-ibs-symptoms-lessen-abdominal-pain/",
                "https://newsnetwork.mayoclinic.org/discussion/mayo-researchers-develop-tool-that-measures-health-of-a-persons-gut-microbiome/"
            ],
            'nih_ncbi': [
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC4566439/",  
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC4290017/", 
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC5433529/", 
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC4528021/", 
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC4425030/", 
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC6682904/",  
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC3577372/", 
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC8995832/", 
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC9455721/"  
            ],
            'precision_nutrition': [
                "https://www.precisionnutrition.com/how-to-improve-gut-health",
                "https://www.precisionnutrition.com/all-about-nutrition-gut-health"
            ]
        }
    
    def extract_healthline_content(self, url: str) -> Dict:
        """Extract structured content from Healthline articles"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text().strip() if title_elem else "No Title"
            
            # Extract main content sections
            content_sections = []
            
            # Find article body - Healthline specific selectors
            article_body = (soup.find('div', class_='content-body') or 
                          soup.find('article') or 
                          soup.find('div', {'data-testid': 'article-content'}))
            
            if article_body:
                # Extract headings and paragraphs
                elements = article_body.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol'])
                
                current_section = {"heading": None, "content": []}
                
                for element in elements:
                    if element.name in ['h2', 'h3', 'h4']:
                        # Save previous section
                        if current_section["heading"] or current_section["content"]:
                            content_sections.append(current_section.copy())
                        
                        # Start new section
                        current_section = {
                            "heading": element.get_text().strip(),
                            "content": []
                        }
                    elif element.name in ['p', 'ul', 'ol']:
                        text = element.get_text().strip()
                        if text and len(text) > 10:  # Filter out very short content
                            current_section["content"].append(text)
                
                # Add last section
                if current_section["heading"] or current_section["content"]:
                    content_sections.append(current_section)
            
            return {
                "source": "healthline",
                "url": url,
                "title": title,
                "sections": content_sections,
                "extraction_status": "success"
            }
            
        except Exception as e:
            return {
                "source": "healthline",
                "url": url,
                "title": None,
                "sections": [],
                "extraction_status": f"error: {str(e)}"
            }
    
    def extract_mayo_clinic_content(self, url: str) -> Dict:
        """Extract structured content from Mayo Clinic articles"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = (soup.find('h1') or 
                         soup.find('title') or 
                         soup.find('h1', class_='page-title'))
            title = title_elem.get_text().strip() if title_elem else "No Title"
            
            content_sections = []
            
            # Mayo Clinic specific content selectors
            article_body = (soup.find('div', class_='content') or 
                          soup.find('main') or 
                          soup.find('article') or
                          soup.find('div', {'role': 'main'}))
            
            if article_body:
                elements = article_body.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol', 'div'])
                
                current_section = {"heading": None, "content": []}
                
                for element in elements:
                    if element.name in ['h2', 'h3', 'h4']:
                        # Save previous section
                        if current_section["heading"] or current_section["content"]:
                            content_sections.append(current_section.copy())
                        
                        # Start new section
                        current_section = {
                            "heading": element.get_text().strip(),
                            "content": []
                        }
                    elif element.name in ['p', 'ul', 'ol']:
                        text = element.get_text().strip()
                        if text and len(text) > 15:
                            current_section["content"].append(text)
                
                # Add last section
                if current_section["heading"] or current_section["content"]:
                    content_sections.append(current_section)
            
            return {
                "source": "mayo_clinic",
                "url": url,
                "title": title,
                "sections": content_sections,
                "extraction_status": "success"
            }
            
        except Exception as e:
            return {
                "source": "mayo_clinic",
                "url": url,
                "title": None,
                "sections": [],
                "extraction_status": f"error: {str(e)}"
            }
    
    def extract_nih_ncbi_content(self, url: str) -> Dict:
        """Extract structured content from NIH/NCBI PMC articles"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = (soup.find('h1', class_='content-title') or 
                         soup.find('title') or
                         soup.find('h1'))
            title = title_elem.get_text().strip() if title_elem else "No Title"
            
            content_sections = []
            
            # PMC specific content selectors
            article_body = (soup.find('div', class_='article-content') or 
                          soup.find('div', {'class': 'article'}) or
                          soup.find('main') or
                          soup.find('article'))
            
            if article_body:
                elements = article_body.find_all(['h2', 'h3', 'h4', 'p', 'div'])
                
                current_section = {"heading": "Abstract", "content": []}  # Start with abstract
                
                for element in elements:
                    if element.name in ['h2', 'h3', 'h4']:
                        # Save previous section
                        if current_section["heading"] or current_section["content"]:
                            content_sections.append(current_section.copy())
                        
                        # Start new section
                        heading_text = element.get_text().strip()
                        current_section = {
                            "heading": heading_text,
                            "content": []
                        }
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if text and len(text) > 20:  # Filter out very short content
                            current_section["content"].append(text)
                
                # Add last section
                if current_section["heading"] or current_section["content"]:
                    content_sections.append(current_section)
            
            return {
                "source": "nih_ncbi",
                "url": url,
                "title": title,
                "sections": content_sections,
                "extraction_status": "success"
            }
            
        except Exception as e:
            return {
                "source": "nih_ncbi",
                "url": url,
                "title": None,
                "sections": [],
                "extraction_status": f"error: {str(e)}"
            }
    
    def extract_precision_nutrition_content(self, url: str) -> Dict:
        """Extract structured content from Precision Nutrition articles"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text().strip() if title_elem else "No Title"
            
            content_sections = []
            
            # Precision Nutrition specific selectors
            article_body = (soup.find('div', class_='post-content') or 
                          soup.find('article') or
                          soup.find('main') or
                          soup.find('div', class_='content'))
            
            if article_body:
                elements = article_body.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol'])
                
                current_section = {"heading": None, "content": []}
                
                for element in elements:
                    if element.name in ['h2', 'h3', 'h4']:
                        # Save previous section
                        if current_section["heading"] or current_section["content"]:
                            content_sections.append(current_section.copy())
                        
                        # Start new section
                        current_section = {
                            "heading": element.get_text().strip(),
                            "content": []
                        }
                    elif element.name in ['p', 'ul', 'ol']:
                        text = element.get_text().strip()
                        if text and len(text) > 15:
                            current_section["content"].append(text)
                
                # Add last section
                if current_section["heading"] or current_section["content"]:
                    content_sections.append(current_section)
            
            return {
                "source": "precision_nutrition",
                "url": url,
                "title": title,
                "sections": content_sections,
                "extraction_status": "success"
            }
            
        except Exception as e:
            return {
                "source": "precision_nutrition",
                "url": url,
                "title": None,
                "sections": [],
                "extraction_status": f"error: {str(e)}"
            }
    
    def extract_all_sources(self, delay: float = 1.0) -> List[Dict]:
        """Extract content from all defined sources"""
        all_extracted_data = []
        
        # Process Healthline sources
        print("Extracting Healthline content...")
        for url in self.data_sources['healthline']:
            print(f"Processing: {url}")
            data = self.extract_healthline_content(url)
            all_extracted_data.append(data)
            time.sleep(delay)
        
        # Process Mayo Clinic sources
        print("\nExtracting Mayo Clinic content...")
        for url in self.data_sources['mayo_clinic']:
            print(f"Processing: {url}")
            data = self.extract_mayo_clinic_content(url)
            all_extracted_data.append(data)
            time.sleep(delay)
        
        # Process NIH/NCBI sources
        print("\nExtracting NIH/NCBI content...")
        for url in self.data_sources['nih_ncbi']:
            print(f"Processing: {url}")
            data = self.extract_nih_ncbi_content(url)
            all_extracted_data.append(data)
            time.sleep(delay)
        
        # Process Precision Nutrition sources
        print("\nExtracting Precision Nutrition content...")
        for url in self.data_sources['precision_nutrition']:
            print(f"Processing: {url}")
            data = self.extract_precision_nutrition_content(url)
            all_extracted_data.append(data)
            time.sleep(delay)
        
        return all_extracted_data
    
    def create_training_qa_pairs(self, extracted_data: List[Dict]) -> List[Dict]:
        """Convert extracted content into Q&A pairs for training"""
        qa_pairs = []
        
        # Define question templates based on gut health topics
        question_templates = {
            "symptoms": [
                "What are the symptoms of {}?",
                "How do I know if I have {}?",
                "What does {} feel like?",
                "What are the signs of {}?"
            ],
            "causes": [
                "What causes {}?",
                "Why do I have {}?",
                "What triggers {}?",
                "What leads to {}?"
            ],
            "treatments": [
                "How do I treat {}?",
                "What helps with {}?",
                "How can I manage {}?",
                "What's the best treatment for {}?"
            ],
            "foods": [
                "What foods help with {}?",
                "What should I eat for {}?",
                "What foods should I avoid with {}?",
                "Which foods are good for {}?"
            ],
            "lifestyle": [
                "How can I improve my {}?",
                "What lifestyle changes help with {}?",
                "How do I prevent {}?",
                "What daily habits support {}?"
            ]
        }
        
        # Extract gut health conditions/topics
        gut_topics = [
            "gut health", "microbiome", "IBS", "bloating", "constipation", 
            "diarrhea", "digestive health", "probiotics", "prebiotics",
            "gut bacteria", "intestinal health", "dysbiosis"
        ]
        
        for article in extracted_data:
            if article["extraction_status"] != "success":
                continue
                
            title = article.get("title", "")
            sections = article.get("sections", [])
            source = article.get("source", "")
            
            # Identify relevant gut health topics in the article
            relevant_topics = [topic for topic in gut_topics 
                             if topic.lower() in title.lower()]
            
            for section in sections:
                heading = section.get("heading") or ""
                content_list = section.get("content", [])

                if not content_list:
                    continue

                content = " ".join(content_list)

                if len(content) < 50:
                    continue

                for topic in relevant_topics:
                    category = "symptoms"
                    if any(word in heading.lower() for word in ["cause", "why", "reason"]):
                        category = "causes"
                    elif any(word in heading.lower() for word in ["treat", "manage", "help", "solution"]):
                        category = "treatments"
                    elif any(word in heading.lower() for word in ["food", "diet", "eat", "nutrition"]):
                        category = "foods"
                    elif any(word in heading.lower() for word in ["lifestyle", "prevent", "improve"]):
                        category = "lifestyle"

                    for template in question_templates[category][:2]:
                        question = template.format(topic)

                        qa_pairs.append({
                            "question": question,
                            "answer": content,
                            "source_url": article.get("url"),
                            "source_name": source,
                            "topic": topic,
                            "category": category,
                            "section_heading": heading,
                            "tone_context": "empathetic_explanation"
                        })
        
        return qa_pairs
    
    def apply_august_ai_tone(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Apply August AI-like tone to the Q&A pairs"""
        
        empathetic_openings = [
            "It's completely understandable to be concerned about this. ",
            "Your symptoms are valid, and here's what might be happening: ",
            "This is more common than you might think. ",
            "You're not alone in experiencing this. ",
            "It's okay — this happens to a lot of people. ",
            "Your concern is valid, and here's what we can look into. "
        ]
        
        # Medical jargon to simple language mapping
        tone_adjustments = {
            "gastrointestinal": "digestive",
            "defecation": "bowel movements", 
            "postprandial": "after eating",
            "etiology": "cause",
            "pathogenesis": "how it develops",
            "symptomatology": "symptoms",
            "therapeutic": "treatment",
            "pharmacological": "medication",
            "ameliorate": "improve",
            "exacerbate": "worsen"
        }
        
        for qa_pair in qa_pairs:
            answer = qa_pair["answer"]
            
            # Add empathetic opening randomly
            if len(answer) > 100:  # Only for substantial answers
                import random
                if random.random() < 0.3:  # 30% chance
                    opening = random.choice(empathetic_openings)
                    answer = opening + answer
            
            # Replace medical jargon with simpler terms
            for medical_term, simple_term in tone_adjustments.items():
                answer = re.sub(r'\b' + medical_term + r'\b', simple_term, answer, flags=re.IGNORECASE)
            
            # Add reassuring language
            if "symptoms" in qa_pair.get("category", ""):
                answer = answer.replace("patients", "people")
                answer = answer.replace("individuals", "people")
            
            qa_pair["answer"] = answer
            qa_pair["tone_applied"] = True
        
        return qa_pairs
    
    def save_data(self, data: List[Dict], filename: str):
        """Save extracted data to JSON file"""
        dir_path = "alldata"
        os.makedirs(dir_path, exist_ok=True) 
        full_path = os.path.join(dir_path, filename)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {full_path}")
    
    def create_training_dataset(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Format Q&A pairs for training (instruction format)"""
        training_data = []
        
        for qa_pair in qa_pairs:
            training_example = {
                "instruction": "You are a compassionate gut health coach. Answer this question with empathy, clarity, and evidence-based guidance. Use accessible language and provide actionable advice.",
                "input": qa_pair["question"],
                "output": qa_pair["answer"],
                "metadata": {
                    "topic": qa_pair.get("topic", ""),
                    "category": qa_pair.get("category", ""),
                    "source": qa_pair.get("source_name", ""),
                    "url": qa_pair.get("source_url", ""),
                    "tone": "august_ai_style"
                }
            }
            training_data.append(training_example)
        
        return training_data

if __name__ == "__main__":
    extractor = GutHealthDataExtractor()

    print("Starting extraction from all sources")
    all_content = extractor.extract_all_sources(delay=1.5)
    extractor.save_data(all_content, "gut_health_raw_data.json")
    
    print("\nCreating Q&A pairs...")
    qa_pairs = extractor.create_training_qa_pairs(all_content)
    
    print("Applying August AI tone...")
    qa_pairs_with_tone = extractor.apply_august_ai_tone(qa_pairs)
    
    training_dataset = extractor.create_training_dataset(qa_pairs_with_tone)
    
    extractor.save_data(qa_pairs_with_tone, "gut_health_qa_pairs.json")
    extractor.save_data(training_dataset, "gut_health_training_dataset.json")
    