import random
from typing import Dict, List

class ScenarioGenerator:
    """Generates random speaking scenarios for aphasia assessment."""
    
    SCENARIOS = [
        {
            "title": "Your Last Vacation",
            "prompt": "Tell me about your last vacation or trip. Where did you go? What did you do there? Who went with you?",
            "keywords": ["vacation", "trip", "travel", "place", "visit"]
        },
        {
            "title": "Your Last Dinner with Family",
            "prompt": "Describe your last dinner time with your family. What did you eat? Who was there? What did you talk about?",
            "keywords": ["dinner", "family", "food", "eat", "meal"]
        },
        {
            "title": "A Typical Morning",
            "prompt": "Tell me about your typical morning routine. What do you do when you wake up? What do you eat for breakfast?",
            "keywords": ["morning", "wake", "breakfast", "routine", "day"]
        },
        {
            "title": "Your Favorite Hobby",
            "prompt": "Describe your favorite hobby or activity. How often do you do it? Why do you enjoy it?",
            "keywords": ["hobby", "activity", "enjoy", "fun", "like"]
        },
        {
            "title": "Last Weekend Activities",
            "prompt": "What did you do last weekend? Tell me about any activities, people you met, or places you visited.",
            "keywords": ["weekend", "saturday", "sunday", "activities", "do"]
        },
        {
            "title": "Your Childhood Home",
            "prompt": "Describe the house or place where you grew up. What did it look like? What are your favorite memories there?",
            "keywords": ["home", "house", "childhood", "grew up", "memories"]
        },
        {
            "title": "A Recent Shopping Trip",
            "prompt": "Tell me about the last time you went shopping. What did you buy? Where did you go? Who was with you?",
            "keywords": ["shopping", "buy", "store", "market", "purchase"]
        },
        {
            "title": "Your Daily Work or Routine",
            "prompt": "Describe what you do during a typical workday or your daily routine. What tasks do you complete?",
            "keywords": ["work", "job", "daily", "routine", "tasks"]
        },
        {
            "title": "A Favorite Movie or Book",
            "prompt": "Tell me about your favorite movie or book. What is it about? Why do you like it? Who are the main characters?",
            "keywords": ["movie", "book", "story", "favorite", "character"]
        },
        {
            "title": "Cooking or Preparing a Meal",
            "prompt": "Describe how you cook or prepare your favorite meal. What ingredients do you need? What are the steps?",
            "keywords": ["cook", "meal", "recipe", "ingredients", "prepare"]
        },
        {
            "title": "A Recent Celebration",
            "prompt": "Tell me about a recent birthday, holiday, or celebration you attended. Who was there? What did you do?",
            "keywords": ["celebration", "birthday", "holiday", "party", "event"]
        },
        {
            "title": "Your Neighborhood",
            "prompt": "Describe your neighborhood or the area where you live. What does it look like? What places are nearby?",
            "keywords": ["neighborhood", "area", "live", "nearby", "place"]
        },
        {
            "title": "Using Your Phone or Computer",
            "prompt": "Tell me about how you use your phone or computer. What do you use it for? What apps or programs do you like?",
            "keywords": ["phone", "computer", "use", "apps", "technology"]
        },
        {
            "title": "A Visit to the Doctor",
            "prompt": "Describe your last visit to a doctor or hospital. Why did you go? What happened during the visit?",
            "keywords": ["doctor", "hospital", "visit", "health", "medical"]
        },
        {
            "title": "Your Favorite Season",
            "prompt": "Tell me about your favorite season of the year. What do you like about it? What activities do you enjoy during that time?",
            "keywords": ["season", "weather", "favorite", "activities", "year"]
        }
    ]
    
    def __init__(self):
        """Initialize scenario generator."""
        self.used_scenarios = []
    
    def get_random_scenario(self, avoid_recent: bool = True) -> Dict:
        """
        Get a random scenario for the user to speak about.
        
        Args:
            avoid_recent: If True, avoid scenarios used recently
            
        Returns:
            Dictionary containing scenario details
        """
        available_scenarios = self.SCENARIOS.copy()
        
        # Remove recently used scenarios if requested
        if avoid_recent and len(self.used_scenarios) > 0:
            available_scenarios = [s for s in available_scenarios 
                                 if s['title'] not in self.used_scenarios]
        
        # If all scenarios have been used, reset
        if len(available_scenarios) == 0:
            available_scenarios = self.SCENARIOS.copy()
            self.used_scenarios = []
        
        # Select random scenario
        scenario = random.choice(available_scenarios)
        
        # Track usage
        self.used_scenarios.append(scenario['title'])
        if len(self.used_scenarios) > 5:  # Keep only last 5
            self.used_scenarios.pop(0)
        
        return scenario
    
    def get_scenario_by_title(self, title: str) -> Dict:
        """
        Get a specific scenario by title.
        
        Args:
            title: Title of the scenario
            
        Returns:
            Dictionary containing scenario details or None if not found
        """
        for scenario in self.SCENARIOS:
            if scenario['title'].lower() == title.lower():
                return scenario
        return None
    
    def list_all_scenarios(self) -> List[str]:
        """
        Get list of all scenario titles.
        
        Returns:
            List of scenario titles
        """
        return [s['title'] for s in self.SCENARIOS]
    
    def display_scenario(self, scenario: Dict, duration: int = 30):
        """
        Display scenario prompt to user.
        
        Args:
            scenario: Scenario dictionary
            duration: Speaking duration in seconds
        """
        print("\n" + "="*60)
        print(f"📋 SPEAKING TASK: {scenario['title']}")
        print("="*60)
        print(f"\n{scenario['prompt']}")
        print(f"\n⏱️  Please speak for approximately {duration} seconds.")
        print("💡 Try to speak naturally and include as much detail as you can.")
        print("="*60 + "\n")
