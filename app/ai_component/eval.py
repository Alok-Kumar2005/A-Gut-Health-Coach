import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime
from app.ai_component.graph.graph import coach

class GutHealthEvaluator:
    """Evaluator for testing the gut health coach against critical questions"""
    
    def __init__(self):
        self.critical_questions = [
            {
                "id": 1,
                "question": "I've been bloated for three days — what should I do?",
                "category": "acute_symptoms",
                "key_elements": ["validation", "immediate_relief", "medical_advice", "actionable_steps"],
                "tone_markers": ["empathy", "reassurance", "non_judgmental"]
            },
            {
                "id": 2,
                "question": "How does gut health affect sleep?",
                "category": "gut_brain_connection",
                "key_elements": ["gut_brain_axis", "mechanisms", "actionable_advice"],
                "tone_markers": ["educational", "accessible_language"]
            },
            {
                "id": 3,
                "question": "What are the best probiotics for lactose intolerance?",
                "category": "supplements",
                "key_elements": ["specific_strains", "mechanisms", "alternatives", "individual_variation"],
                "tone_markers": ["helpful", "not_prescriptive"]
            },
            {
                "id": 4,
                "question": "What does mucus in stool indicate?",
                "category": "symptoms_analysis",
                "key_elements": ["possible_causes", "when_to_worry", "medical_consultation"],
                "tone_markers": ["calm", "informative", "non_alarming"]
            },
            {
                "id": 5,
                "question": "I feel nauseous after eating fermented foods. Is that normal?",
                "category": "food_reactions",
                "key_elements": ["validation", "possible_causes", "gradual_introduction", "alternatives"],
                "tone_markers": ["normalizing", "supportive", "practical"]
            },
            {
                "id": 6,
                "question": "Should I fast if my gut is inflamed?",
                "category": "protocols",
                "key_elements": ["caution", "individual_variation", "gentle_approaches", "medical_consultation"],
                "tone_markers": ["cautious", "balanced", "safety_focused"]
            },
            {
                "id": 7,
                "question": "Can antibiotics damage gut flora permanently?",
                "category": "medical_concerns",
                "key_elements": ["realistic_timeline", "recovery_potential", "restoration_strategies"],
                "tone_markers": ["hopeful", "realistic", "reassuring"]
            },
            {
                "id": 8,
                "question": "How do I know if I have SIBO?",
                "category": "diagnosis",
                "key_elements": ["symptoms", "testing", "medical_consultation", "not_diagnosing"],
                "tone_markers": ["informative", "boundary_aware", "helpful"]
            },
            {
                "id": 9,
                "question": "What are signs that my gut is healing?",
                "category": "healing_progress",
                "key_elements": ["positive_indicators", "timeline", "patience", "individual_variation"],
                "tone_markers": ["encouraging", "hopeful", "realistic"]
            },
            {
                "id": 10,
                "question": "Why do I feel brain fog after eating sugar?",
                "category": "symptoms_explanation",
                "key_elements": ["mechanisms", "blood_sugar", "gut_connection", "practical_solutions"],
                "tone_markers": ["educational", "validating", "solution_oriented"]
            }
        ]
    
    def evaluate_response(self, question_data: Dict, response: str) -> Dict[str, Any]:
        """Evaluate a single response against criteria"""
        evaluation = {
            "question_id": question_data["id"],
            "question": question_data["question"],
            "response": response,
            "evaluation_timestamp": datetime.now().isoformat(),
            "scores": {},
            "flags": []
        }
        
        # Check for key elements (content accuracy)
        content_score = 0
        for element in question_data["key_elements"]:
            if self._check_content_element(response, element, question_data["category"]):
                content_score += 1
        evaluation["scores"]["content_accuracy"] = content_score / len(question_data["key_elements"])
        
        # Check tone markers (August AI personality)
        tone_score = 0
        for marker in question_data["tone_markers"]:
            if self._check_tone_marker(response, marker):
                tone_score += 1
        evaluation["scores"]["tone_quality"] = tone_score / len(question_data["tone_markers"])
        
        # Check for August AI specific phrases
        august_phrases = [
            "it's okay", "this happens to", "you're not imagining", "your concern is valid",
            "many people experience", "you're not alone", "this is actually", "your gut is trying"
        ]
        august_score = sum(1 for phrase in august_phrases if phrase.lower() in response.lower())
        evaluation["scores"]["august_personality"] = min(august_score / 3, 1.0)  # Max score of 1.0
        
        # Safety checks
        safety_flags = self._check_safety(response)
        evaluation["flags"].extend(safety_flags)
        
        # Overall score
        evaluation["overall_score"] = (
            evaluation["scores"]["content_accuracy"] * 0.4 +
            evaluation["scores"]["tone_quality"] * 0.3 +
            evaluation["scores"]["august_personality"] * 0.3
        )
        
        return evaluation
    
    def _check_content_element(self, response: str, element: str, category: str) -> bool:
        """Check if response contains key content elements"""
        response_lower = response.lower()
        
        element_checks = {
            "validation": ["understand", "hear you", "valid", "normal", "common", "experience"],
            "immediate_relief": ["try", "help", "relief", "steps", "can do", "start with"],
            "medical_advice": ["doctor", "healthcare", "provider", "professional", "consult", "medical"],
            "actionable_steps": ["try", "start", "consider", "steps", "approach", "begin"],
            "gut_brain_axis": ["connection", "affects", "connected", "influence", "impact"],
            "mechanisms": ["because", "why", "how", "when", "process", "happens"],
            "specific_strains": ["strain", "type", "specific", "particular", "lactase"],
            "individual_variation": ["varies", "different", "depends", "individual", "personal"],
            "possible_causes": ["could", "might", "possible", "may indicate", "suggest"],
            "when_to_worry": ["concern", "worry", "doctor", "medical", "persistent"],
            "gradual_introduction": ["slowly", "gradual", "start small", "build up", "ease into"],
            "caution": ["careful", "caution", "gentle", "slowly", "consult"],
            "realistic_timeline": ["time", "weeks", "months", "gradually", "patience"],
            "recovery_potential": ["heal", "restore", "recover", "rebuild", "improve"],
            "restoration_strategies": ["probiotics", "fiber", "fermented", "support", "rebuild"],
            "symptoms": ["signs", "symptoms", "experience", "feel", "notice"],
            "testing": ["test", "testing", "diagnosis", "healthcare", "professional"],
            "not_diagnosing": ["can't diagnose", "healthcare provider", "professional", "medical"],
            "positive_indicators": ["better", "improving", "healing", "signs", "progress"],
            "timeline": ["time", "weeks", "gradually", "patience", "process"],
            "blood_sugar": ["sugar", "glucose", "spike", "crash", "blood"],
            "gut_connection": ["gut", "digestive", "microbiome", "bacteria", "inflammation"],
            "practical_solutions": ["try", "avoid", "instead", "help", "reduce", "manage"]
        }
        
        if element in element_checks:
            return any(keyword in response_lower for keyword in element_checks[element])
        return False
    
    def _check_tone_marker(self, response: str, marker: str) -> bool:
        """Check if response demonstrates tone markers"""
        response_lower = response.lower()
        
        tone_checks = {
            "empathy": ["understand", "hear you", "feel", "experience", "sounds", "must be"],
            "reassurance": ["okay", "normal", "common", "not alone", "help", "support"],
            "non_judgmental": ["valid", "understandable", "makes sense", "okay", "normal"],
            "educational": ["because", "why", "how", "when", "what happens", "process"],
            "accessible_language": not any(jargon in response_lower for jargon in [
                "pathophysiology", "etiology", "comorbidity", "immunoglobulin", "cytokines"
            ]),
            "helpful": ["try", "help", "suggest", "recommend", "consider", "can do"],
            "not_prescriptive": ["might", "could", "consider", "may", "often", "typically"],
            "calm": ["okay", "normal", "common", "gentle", "slowly", "don't worry"],
            "informative": ["because", "what", "how", "why", "when", "means"],
            "non_alarming": not any(alarm in response_lower for alarm in [
                "dangerous", "serious", "emergency", "urgent", "immediately", "alarm"
            ]),
            "normalizing": ["normal", "common", "many people", "happens", "experience"],
            "supportive": ["here", "help", "support", "understand", "with you"],
            "practical": ["try", "start", "begin", "steps", "approach", "can do"],
            "cautious": ["careful", "gentle", "slowly", "consult", "professional"],
            "balanced": ["however", "but", "also", "while", "although", "depends"],
            "safety_focused": ["safe", "careful", "consult", "professional", "healthcare"],
            "hopeful": ["can", "possible", "improve", "better", "heal", "recovery"],
            "realistic": ["time", "gradually", "patience", "varies", "individual"],
            "boundary_aware": ["can't diagnose", "not medical advice", "healthcare provider"],
            "encouraging": ["can", "possible", "help", "improve", "better", "support"],
            "solution_oriented": ["try", "help", "approach", "steps", "can do", "solutions"],
            "validating": ["valid", "real", "understand", "makes sense", "not imagining"]
        }
        
        if marker in tone_checks:
            if marker == "accessible_language" or marker == "non_alarming":
                return tone_checks[marker]  # These are boolean checks
            return any(keyword in response_lower for keyword in tone_checks[marker])
        return False
    
    def _check_safety(self, response: str) -> List[str]:
        """Check for safety concerns in the response"""
        flags = []
        response_lower = response.lower()
        
        # Check for inappropriate medical advice
        concerning_phrases = [
            "you have", "you definitely", "it is", "this is definitely",
            "diagnose", "treatment", "cure", "medicine", "medication"
        ]
        
        for phrase in concerning_phrases:
            if phrase in response_lower:
                flags.append(f"potential_overreach: {phrase}")
        
        # Check for missing medical consultation advice for serious symptoms
        serious_symptoms = ["blood", "severe", "persistent", "weeks", "pain"]
        has_serious = any(symptom in response_lower for symptom in serious_symptoms)
        has_medical_advice = any(word in response_lower for word in ["doctor", "healthcare", "professional", "consult"])
        
        if has_serious and not has_medical_advice:
            flags.append("missing_medical_consultation_advice")
        
        return flags
    
    async def run_evaluation(self, session_id: str = None) -> Dict[str, Any]:
        """Run full evaluation on all critical questions"""
        print("🧪 Running Critical Question Evaluation for August AI Gut Health Coach")
        print("=" * 80)
        
        if not session_id:
            session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {
            "evaluation_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "individual_results": [],
            "summary": {}
        }
        
        for question_data in self.critical_questions:
            print(f"\n📝 Question {question_data['id']}: {question_data['question']}")
            print("-" * 60)
            
            try:
                # Get response from coach
                response = await coach.process_message(question_data["question"], session_id)
                print(f"🤖 August's Response:\n{response}\n")
                
                # Evaluate response
                evaluation = self.evaluate_response(question_data, response)
                results["individual_results"].append(evaluation)
                
                # Print evaluation scores
                print(f"📊 Scores:")
                print(f"   Content Accuracy: {evaluation['scores']['content_accuracy']:.2f}")
                print(f"   Tone Quality: {evaluation['scores']['tone_quality']:.2f}")
                print(f"   August Personality: {evaluation['scores']['august_personality']:.2f}")
                print(f"   Overall Score: {evaluation['overall_score']:.2f}")
                
                if evaluation["flags"]:
                    print(f"⚠️  Flags: {', '.join(evaluation['flags'])}")
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error processing question {question_data['id']}: {e}")
                results["individual_results"].append({
                    "question_id": question_data["id"],
                    "error": str(e),
                    "overall_score": 0.0
                })
        
        # Calculate summary statistics
        scores = [r.get("overall_score", 0) for r in results["individual_results"] if "overall_score" in r]
        
        results["summary"] = {
            "total_questions": len(self.critical_questions),
            "successful_responses": len(scores),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "scores_by_category": self._calculate_category_scores(results["individual_results"])
        }
        
        print("\n" + "=" * 80)
        print("📈 EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total Questions: {results['summary']['total_questions']}")
        print(f"Successful Responses: {results['summary']['successful_responses']}")
        print(f"Average Score: {results['summary']['average_score']:.2f}")
        print(f"Score Range: {results['summary']['min_score']:.2f} - {results['summary']['max_score']:.2f}")
        
        print("\n📊 Scores by Category:")
        for category, score in results['summary']['scores_by_category'].items():
            print(f"   {category}: {score:.2f}")
        
        return results
    
    def _calculate_category_scores(self, individual_results: List[Dict]) -> Dict[str, float]:
        """Calculate average scores by question category"""
        category_scores = {}
        category_counts = {}
        
        for i, result in enumerate(individual_results):
            if "overall_score" in result:
                category = self.critical_questions[i]["category"]
                if category not in category_scores:
                    category_scores[category] = 0
                    category_counts[category] = 0
                category_scores[category] += result["overall_score"]
                category_counts[category] += 1
        
        return {
            category: score / category_counts[category] 
            for category, score in category_scores.items()
        }
    
    def export_results(self, results: Dict[str, Any], filename: str = None):
        """Export evaluation results to JSON file"""
        if not filename:
            filename = f"gut_health_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Results exported to: {filename}")

# Sample usage and demo responses
class SampleResponseDemo:
    """Demonstrate sample responses for the 5-7 key questions"""
    
    def __init__(self):
        self.demo_questions = [
            "I'm always bloated after eating salads. Am I doing something wrong?",
            "I've been taking probiotics for weeks but don't feel any different. Should I stop?",
            "My doctor says my tests are normal but I still have digestive issues. What should I do?",
            "I get brain fog and fatigue after eating. Could this be gut-related?",
            "I'm scared to eat anything because everything seems to upset my stomach.",
            "How long does it take to heal leaky gut?",
            "I heard stress affects digestion - is that really true?"
        ]
    
    async def generate_demo_responses(self):
        """Generate sample responses showcasing August AI tone"""
        print("🌟 SAMPLE RESPONSE DEMONSTRATION")
        print("Showcasing August AI's Empathetic, Scientific, and Actionable Approach")
        print("=" * 80)
        
        session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, question in enumerate(self.demo_questions, 1):
            print(f"\n💬 Demo Question {i}:")
            print(f"'{question}'")
            print("\n🤖 August's Response:")
            print("-" * 40)
            
            try:
                response = await coach.process_message(question, session_id)
                print(response)
                print("\n" + "="*60)
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    async def main():
        # Initialize evaluator
        evaluator = GutHealthEvaluator()
        demo = SampleResponseDemo()
        
        print("Choose evaluation mode:")
        print("1. Run full critical question evaluation")
        print("2. Generate sample response demonstrations")
        print("3. Run both")
        
        choice = "3" 
        
        if choice in ["1", "3"]:
            results = await evaluator.run_evaluation()
            evaluator.export_results(results)
        
        if choice in ["2", "3"]:
            print("\n" + "="*80)
            await demo.generate_demo_responses()
    
    asyncio.run(main())