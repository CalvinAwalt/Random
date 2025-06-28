import random
import time
from datetime import datetime, timedelta
import webbrowser
import sqlite3
from collections import defaultdict
import textwrap

class StudyAssistant:
    def __init__(self):
        self.notes = defaultdict(list)
        self.flashcards = {}
        self.conn = sqlite3.connect('study_data.db')
        self.create_tables()
        
    def create_tables(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS study_sessions
                         (id INTEGER PRIMARY KEY, topic TEXT, start_time TEXT, 
                         duration INTEGER, notes TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS flashcards
                         (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, 
                         topic TEXT, last_reviewed TEXT, difficulty INTEGER)''')
        self.conn.commit()

    def pomodoro_timer(self, work_min=25, break_min=5):
        """Pomodoro technique timer"""
        print(f"\n🍅 Starting Pomodoro: {work_min} min work, {break_min} min break")
        while True:
            # Work period
            print(f"\nWORK TIME - {datetime.now().strftime('%H:%M')}")
            self.countdown(work_min * 60)
            print("\nTime's up! Take a break.")
            
            # Break period
            print(f"\nBREAK TIME - {datetime.now().strftime('%H:%M')}")
            self.countdown(break_min * 60)
            print("\nBreak over! Ready for another session?")
            
            if input("Continue? (y/n): ").lower() != 'y':
                break

    def countdown(self, seconds):
        """Countdown timer"""
        for remaining in range(seconds, 0, -1):
            mins, secs = divmod(remaining, 60)
            print(f"\r⏳ {mins:02d}:{secs:02d} remaining", end="", flush=True)
            time.sleep(1)
        print("\r" + " " * 30 + "\r", end="")

    def take_notes(self, topic):
        """Digital note-taking system"""
        print(f"\n📝 Taking notes on: {topic}")
        print("Type your notes below. Enter 'save' on a new line when done:")
        note_lines = []
        while True:
            line = input()
            if line.lower() == 'save':
                break
            note_lines.append(line)
        
        full_note = '\n'.join(note_lines)
        self.notes[topic].append(full_note)
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO study_sessions 
                          (topic, start_time, duration, notes) 
                          VALUES (?, ?, ?, ?)''',
                       (topic, datetime.now().isoformat(), 
                        len(note_lines) * 2, full_note))  # Estimate 2 sec per line
        self.conn.commit()
        print(f"✓ Saved {len(note_lines)} lines of notes on {topic}")

    def add_flashcard(self):
        """Create a flashcard"""
        question = input("Enter question: ")
        answer = input("Enter answer: ")
        topic = input("Enter topic (optional): ") or "General"
        
        self.flashcards[question] = (answer, topic)
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO flashcards 
                          (question, answer, topic, last_reviewed, difficulty) 
                          VALUES (?, ?, ?, ?, ?)''',
                       (question, answer, topic, datetime.now().isoformat(), 3))
        self.conn.commit()
        print("✓ Flashcard added!")

    def quiz_flashcards(self, topic=None):
        """Quiz yourself with flashcards"""
        if topic:
            cards = {q:(a,t) for q,(a,t) in self.flashcards.items() if t == topic}
        else:
            cards = self.flashcards
            
        if not cards:
            print("No flashcards found!" if topic else "No flashcards found for this topic!")
            return
            
        questions = list(cards.keys())
        random.shuffle(questions)
        
        correct = 0
        for i, question in enumerate(questions, 1):
            answer, _ = cards[question]
            print(f"\nCard {i}/{len(questions)}")
            print(f"Q: {question}")
            input("Press Enter to see answer...")
            print(f"A: {answer}")
            
            while True:
                rating = input("Did you get it right? (1-5 where 5=easy): ")
                if rating.isdigit() and 1 <= int(rating) <= 5:
                    break
                print("Please enter a number 1-5")
            
            if int(rating) >= 3:
                correct += 1
        
        print(f"\nScore: {correct}/{len(questions)} ({correct/len(questions)*100:.1f}%)")

    def focus_mode(self):
        """Block distracting websites for a set time"""
        sites_to_block = [
            "facebook.com", "twitter.com", "reddit.com", 
            "youtube.com", "instagram.com", "tiktok.com"
        ]
        
        duration = int(input("Enter focus duration in minutes: "))
        print(f"\n🚀 Starting focus mode for {duration} minutes")
        print("These sites will be blocked:")
        print(", ".join(sites_to_block))
        
        # In a real implementation, you would modify the hosts file
        # This is a simulation for demonstration
        start_time = time.time()
        while time.time() - start_time < duration * 60:
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            print(f"\rFocus time elapsed: {mins:02d}:{secs:02d}", end="")
        
        print("\n\nFocus session complete! 🎉")

    def study_stats(self):
        """Show study statistics"""
        cursor = self.conn.cursor()
        
        # Total study time
        cursor.execute("SELECT SUM(duration) FROM study_sessions")
        total_min = cursor.fetchone()[0] or 0
        print(f"\n📊 Total Study Time: {total_min} minutes")
        
        # By topic
        cursor.execute('''SELECT topic, SUM(duration) 
                          FROM study_sessions 
                          GROUP BY topic 
                          ORDER BY SUM(duration) DESC''')
        print("\n⏱️ Time by Topic:")
        for topic, duration in cursor.fetchall():
            print(f"- {topic}: {duration} minutes")
        
        # Flashcard stats
        cursor.execute("SELECT COUNT(*) FROM flashcards")
        total_cards = cursor.fetchone()[0]
        print(f"\n📇 Total Flashcards: {total_cards}")

    def spaced_repetition_schedule(self):
        """Recommend what to study based on spaced repetition"""
        cursor = self.conn.cursor()
        
        # Get flashcards sorted by difficulty and last reviewed
        cursor.execute('''SELECT question, topic, difficulty, last_reviewed 
                          FROM flashcards 
                          ORDER BY difficulty ASC, last_reviewed ASC
                          LIMIT 5''')
        
        print("\n🔁 Recommended for Review (Spaced Repetition):")
        for i, (question, topic, difficulty, last_reviewed) in enumerate(cursor.fetchall(), 1):
            last_date = datetime.fromisoformat(last_reviewed).strftime('%Y-%m-%d')
            print(f"{i}. {topic}: {question[:50]}... (difficulty: {difficulty}/5, last: {last_date})")

    def run(self):
        """Main menu interface"""
        print("\n📚 Python Study Assistant 📚")
        while True:
            print("\nMain Menu:")
            print("1. Pomodoro Timer")
            print("2. Take Notes")
            print("3. Add Flashcard")
            print("4. Quiz Flashcards")
            print("5. Focus Mode")
            print("6. Study Statistics")
            print("7. Spaced Repetition Schedule")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ")
            
            if choice == '1':
                self.pomodoro_timer()
            elif choice == '2':
                topic = input("Enter topic: ")
                self.take_notes(topic)
            elif choice == '3':
                self.add_flashcard()
            elif choice == '4':
                topic = input("Enter topic to quiz (or leave blank for all): ")
                self.quiz_flashcards(topic if topic else None)
            elif choice == '5':
                self.focus_mode()
            elif choice == '6':
                self.study_stats()
            elif choice == '7':
                self.spaced_repetition_schedule()
            elif choice == '8':
                print("\nGoodbye and happy studying! 🎓")
                self.conn.close()
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    assistant = StudyAssistant()
    assistant.run()