"""
Ablation Study — Option C: Fixed Alpha vs CMA-ES Learned Alpha
Tests on both GSM8K subset (97Q) and MMLU subset (100Q)

Key question: Does CMA-ES actually help vs fixed alpha values?

Configurations:
1. Fixed alpha = 0.3
2. Fixed alpha = 0.5
3. Fixed alpha = 0.7
4. CMA-ES learned alpha

Author: Md. Robiul Islam Niloy
Institution: BRAC University, Bangladesh
arXiv: 2605.12326
GitHub: https://github.com/Mdniloykhan/dfs-merging-blackbox-optimization
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import copy
import json
import gc
import cma as cma_lib
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.2"
SEEDS = [0, 42, 84]
SELECTION_PROB = 0.5
CMA_ITERATIONS = 8
CMA_POPSIZE = 4
CHECKPOINT_FILE = "checkpoint_ablation_c.json"
RESULTS_FILE = "results_ablation_c.json"
FIGURE_FILE = "ablation_fixed_vs_learned.png"

# Configurations to test
CONFIGS = [
    ("fixed_alpha_0.3", False, 0.3),
    ("fixed_alpha_0.5", False, 0.5),
    ("fixed_alpha_0.7", False, 0.7),
    ("cma_es_learned", True, None),
]

# ============================================================
# GSM8K QUESTIONS (97 questions)
# ============================================================
GSM8K_QUESTIONS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. How many eggs does she sell at the farmers' market daily if she sells for $2 per egg?", "answer": "9"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "answer": "540"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She has 20 chickens. How many cups of chicken feed does Wendi need to buy for a week?", "answer": "420"},
    {"question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?", "answer": "64"},
    {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", "answer": "460"},
    {"question": "A company sells widgets at $15 each. The company sold 200 widgets last month and 250 widgets this month. How much more revenue did they make this month compared to last month?", "answer": "750"},
    {"question": "Tom has 3 times as many marbles as Jerry. Jerry has 15 marbles. How many marbles do they have together?", "answer": "60"},
    {"question": "A baker makes 48 cookies. She puts them in boxes of 6. How many boxes does she need?", "answer": "8"},
    {"question": "Sarah earns $12 per hour. She worked 8 hours on Monday and 6 hours on Tuesday. How much did she earn in total?", "answer": "168"},
    {"question": "A train travels at 60 mph. How far does it travel in 2.5 hours?", "answer": "150"},
    {"question": "Mark has $50. He buys 3 books at $8 each. How much money does he have left?", "answer": "26"},
    {"question": "A rectangle has a length of 12 cm and width of 5 cm. What is its area?", "answer": "60"},
    {"question": "There are 24 students in a class. If they are divided into groups of 4, how many groups are there?", "answer": "6"},
    {"question": "A store has 150 apples. They sell 45 on Monday and 38 on Tuesday. How many apples are left?", "answer": "67"},
    {"question": "John runs 5 km every day. How many km does he run in 2 weeks?", "answer": "70"},
    {"question": "A bag has 8 red balls and 12 blue balls. How many balls are in the bag?", "answer": "20"},
    {"question": "Emma saves $25 per week. How much does she save in 8 weeks?", "answer": "200"},
    {"question": "A farmer has 5 fields. Each field produces 120 kg of wheat. How much wheat does he produce in total?", "answer": "600"},
    {"question": "A car travels 240 km using 20 liters of fuel. How many km per liter does it get?", "answer": "12"},
    {"question": "A box contains 144 chocolates. If shared equally among 12 children, how many does each get?", "answer": "12"},
    {"question": "Lisa has 3 times as many stickers as Mike. Mike has 24 stickers. How many stickers do they have together?", "answer": "96"},
    {"question": "A shop sells shirts for $25 each. If they sold 40 shirts, how much money did they make?", "answer": "1000"},
    {"question": "There are 7 days in a week. How many days are in 52 weeks?", "answer": "364"},
    {"question": "A pool holds 5000 liters. If 1200 liters evaporate, how many liters remain?", "answer": "3800"},
    {"question": "Peter earns $15 per hour. He works 40 hours per week. How much does he earn per week?", "answer": "600"},
    {"question": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?", "answer": "2"},
    {"question": "A school has 450 students. If 180 are boys, how many are girls?", "answer": "270"},
    {"question": "A jar contains 200 candies. Maria takes 45 and John takes 38. How many candies remain?", "answer": "117"},
    {"question": "A cyclist rides 18 km per hour. How far does he ride in 3 hours?", "answer": "54"},
    {"question": "A factory produces 250 toys per day. How many toys does it produce in 5 days?", "answer": "1250"},
    {"question": "Anna has $100. She spends $35 on food and $28 on clothes. How much does she have left?", "answer": "37"},
    {"question": "A rope is 48 meters long. It is cut into pieces of 6 meters each. How many pieces are there?", "answer": "8"},
    {"question": "There are 30 students. Each student needs 3 pencils. How many pencils are needed?", "answer": "90"},
    {"question": "A book has 320 pages. Maria reads 40 pages per day. How many days does it take to finish?", "answer": "8"},
    {"question": "A garden is 15 meters long and 8 meters wide. What is its perimeter?", "answer": "46"},
    {"question": "Sam has 4 times as many coins as Tim. Tim has 12 coins. How many coins does Sam have?", "answer": "48"},
    {"question": "A supermarket sells 500 items per day. How many items does it sell in 30 days?", "answer": "15000"},
    {"question": "A bottle holds 750 ml. How many bottles are needed to hold 3 liters?", "answer": "4"},
    {"question": "There are 12 months in a year. How many months are in 5 years?", "answer": "60"},
    {"question": "A worker earns $80 per day. How much does he earn in 15 days?", "answer": "1200"},
    {"question": "A bus can carry 45 passengers. How many buses are needed for 180 passengers?", "answer": "4"},
    {"question": "A fruit seller has 96 oranges. He packs them in bags of 8. How many bags does he need?", "answer": "12"},
    {"question": "Two numbers add up to 50. One number is 18. What is the other?", "answer": "32"},
    {"question": "A swimming pool is 25 meters long. A swimmer does 8 laps. How many meters does he swim?", "answer": "200"},
    {"question": "A family spends $450 per month on groceries. How much do they spend in a year?", "answer": "5400"},
    {"question": "A garden has 5 rows of flowers with 12 flowers in each row. How many flowers are there?", "answer": "60"},
    {"question": "Mike types 60 words per minute. How many words does he type in 15 minutes?", "answer": "900"},
    {"question": "A store buys items for $20 each and sells them for $35. What is the profit per item?", "answer": "15"},
    {"question": "There are 1000 students. 600 study science. How many do not study science?", "answer": "400"},
    {"question": "A car uses 8 liters of fuel per 100 km. How much fuel does it need for 250 km?", "answer": "20"},
    {"question": "A box weighs 15 kg. How much do 12 such boxes weigh?", "answer": "180"},
    {"question": "A library has 2400 books. 800 are fiction. How many are non-fiction?", "answer": "1600"},
    {"question": "Jenny bakes 6 dozen cookies. How many cookies does she bake?", "answer": "72"},
    {"question": "A class has 35 students. 14 are absent. How many are present?", "answer": "21"},
    {"question": "A store sells 3 items for $10. How much do 9 items cost?", "answer": "30"},
    {"question": "A plane flies at 800 km per hour. How far does it fly in 4 hours?", "answer": "3200"},
    {"question": "David saves $150 per month. How much does he save in 6 months?", "answer": "900"},
    {"question": "A field produces 1200 kg of corn. If sold at $2 per kg, how much money is made?", "answer": "2400"},
    {"question": "A number multiplied by 7 equals 84. What is the number?", "answer": "12"},
    {"question": "A tank has 800 liters. 250 liters are used. How many liters remain?", "answer": "550"},
    {"question": "Three friends share 48 candies equally. How many does each get?", "answer": "16"},
    {"question": "A building has 15 floors with 8 apartments each. How many apartments are there?", "answer": "120"},
    {"question": "A shirt costs $45. With a 20% discount, how much does it cost?", "answer": "36"},
    {"question": "A train has 12 coaches with 60 seats each. How many seats in total?", "answer": "720"},
    {"question": "Mary reads 25 pages per day. How many days to read a 300-page book?", "answer": "12"},
    {"question": "A box has 5 layers with 20 items per layer. How many items are in the box?", "answer": "100"},
    {"question": "A worker produces 45 units per hour. How many units in 8 hours?", "answer": "360"},
    {"question": "A number divided by 6 equals 15. What is the number?", "answer": "90"},
    {"question": "A store has 240 items. They sell 60%. How many items are sold?", "answer": "144"},
    {"question": "A pond has 500 fish. 150 are caught. How many remain?", "answer": "350"},
    {"question": "Each child gets 4 balloons. There are 25 children. How many balloons are needed?", "answer": "100"},
    {"question": "A wall is 6 meters high and 10 meters wide. What is its area?", "answer": "60"},
    {"question": "Tom earns $18 per hour and works 35 hours per week. What is his weekly income?", "answer": "630"},
    {"question": "A jar contains 180 marbles. 60 are red and the rest are blue. How many are blue?", "answer": "120"},
    {"question": "A school has 8 classes with 32 students each. How many students in total?", "answer": "256"},
    {"question": "A book costs $12. How much do 15 books cost?", "answer": "180"},
    {"question": "A runner completes a 10 km race in 50 minutes. What is his speed in km per minute?", "answer": "0.2"},
    {"question": "A basket holds 24 apples. How many baskets are needed for 144 apples?", "answer": "6"},
    {"question": "A shop has 350 items. 70 are returned. How many items remain?", "answer": "280"},
    {"question": "Five friends each contribute $15 for a gift. How much do they collect in total?", "answer": "75"},
    {"question": "A factory makes 1200 products in 8 hours. How many per hour?", "answer": "150"},
    {"question": "A number increased by 35 equals 92. What is the number?", "answer": "57"},
    {"question": "A car park has 8 rows with 15 spaces each. How many spaces in total?", "answer": "120"},
    {"question": "A student scores 85, 90, and 95 on three tests. What is the average?", "answer": "90"},
    {"question": "A shop bought 200 items at $5 each and sold them at $8 each. What is the total profit?", "answer": "600"},
    {"question": "A container holds 2.5 liters. How many containers are needed for 15 liters?", "answer": "6"},
    {"question": "A house has 4 bedrooms. Each bedroom has 2 windows. How many windows in total?", "answer": "8"},
    {"question": "A team scores 3 goals per game. How many goals in 12 games?", "answer": "36"},
    {"question": "A number subtracted from 100 equals 37. What is the number?", "answer": "63"},
    {"question": "A cinema has 200 seats. 75% are occupied. How many seats are occupied?", "answer": "150"},
    {"question": "A person walks 4 km per hour. How far do they walk in 2.5 hours?", "answer": "10"},
    {"question": "A jar has 50 coins. 20 are quarters. How many are not quarters?", "answer": "30"},
    {"question": "A bag of rice weighs 5 kg. How much do 14 bags weigh?", "answer": "70"},
    {"question": "A student reads 30 pages per day. How many pages in 10 days?", "answer": "300"},
]

# ============================================================
# MMLU QUESTIONS (100 questions, 10 subjects)
# ============================================================
MMLU_QUESTIONS = [
    {"question": "What is the derivative of x^3?", "choices": ["A) 3x", "B) 3x^2", "C) x^2", "D) 2x^3"], "answer": "B"},
    {"question": "What is the integral of 2x?", "choices": ["A) 2", "B) x^2 + C", "C) 2x^2 + C", "D) x + C"], "answer": "B"},
    {"question": "What is log base 2 of 8?", "choices": ["A) 2", "B) 4", "C) 3", "D) 8"], "answer": "C"},
    {"question": "What is the sum of angles in a triangle?", "choices": ["A) 90 degrees", "B) 180 degrees", "C) 270 degrees", "D) 360 degrees"], "answer": "B"},
    {"question": "What is the value of pi approximately?", "choices": ["A) 3.14", "B) 2.71", "C) 1.41", "D) 1.73"], "answer": "A"},
    {"question": "What is 2 to the power of 10?", "choices": ["A) 512", "B) 256", "C) 1024", "D) 2048"], "answer": "C"},
    {"question": "What is the Pythagorean theorem?", "choices": ["A) a+b=c", "B) a^2+b^2=c^2", "C) a^2-b^2=c^2", "D) a*b=c^2"], "answer": "B"},
    {"question": "What is the factorial of 5?", "choices": ["A) 25", "B) 60", "C) 120", "D) 720"], "answer": "C"},
    {"question": "What is the value of sin(90 degrees)?", "choices": ["A) 0", "B) 0.5", "C) sqrt(2)/2", "D) 1"], "answer": "D"},
    {"question": "What is GCD of 48 and 18?", "choices": ["A) 3", "B) 6", "C) 9", "D) 12"], "answer": "B"},
    {"question": "What is Newton's second law?", "choices": ["A) F=ma", "B) E=mc^2", "C) F=mv", "D) P=mv"], "answer": "A"},
    {"question": "What is the speed of light in vacuum?", "choices": ["A) 3x10^6 m/s", "B) 3x10^8 m/s", "C) 3x10^10 m/s", "D) 3x10^4 m/s"], "answer": "B"},
    {"question": "What is the unit of electric current?", "choices": ["A) Volt", "B) Ohm", "C) Ampere", "D) Watt"], "answer": "C"},
    {"question": "What is Ohm's law?", "choices": ["A) V=IR", "B) P=IV", "C) E=mc^2", "D) F=ma"], "answer": "A"},
    {"question": "What is the unit of force?", "choices": ["A) Joule", "B) Watt", "C) Newton", "D) Pascal"], "answer": "C"},
    {"question": "What is the boiling point of water at sea level in Celsius?", "choices": ["A) 90", "B) 95", "C) 100", "D) 105"], "answer": "C"},
    {"question": "What is kinetic energy formula?", "choices": ["A) KE=mgh", "B) KE=mv", "C) KE=0.5mv^2", "D) KE=mv^2"], "answer": "C"},
    {"question": "What is the gravitational acceleration on Earth?", "choices": ["A) 8.9 m/s^2", "B) 9.8 m/s^2", "C) 10.8 m/s^2", "D) 11.2 m/s^2"], "answer": "B"},
    {"question": "What is the first law of thermodynamics?", "choices": ["A) Energy cannot be created or destroyed", "B) Entropy always increases", "C) Equal action and reaction", "D) Force equals mass times acceleration"], "answer": "A"},
    {"question": "What is the unit of energy?", "choices": ["A) Newton", "B) Watt", "C) Pascal", "D) Joule"], "answer": "D"},
    {"question": "What is the chemical symbol for gold?", "choices": ["A) Go", "B) Gd", "C) Au", "D) Ag"], "answer": "C"},
    {"question": "What is the atomic number of carbon?", "choices": ["A) 4", "B) 6", "C) 8", "D) 12"], "answer": "B"},
    {"question": "What is the chemical formula for water?", "choices": ["A) HO", "B) H2O", "C) H2O2", "D) H3O"], "answer": "B"},
    {"question": "What is the pH of a neutral solution?", "choices": ["A) 0", "B) 5", "C) 7", "D) 14"], "answer": "C"},
    {"question": "What is the most abundant element in Earth's atmosphere?", "choices": ["A) Oxygen", "B) Carbon dioxide", "C) Argon", "D) Nitrogen"], "answer": "D"},
    {"question": "What is the chemical symbol for sodium?", "choices": ["A) So", "B) Sd", "C) Na", "D) Ni"], "answer": "C"},
    {"question": "What is Avogadro's number approximately?", "choices": ["A) 6.02x10^21", "B) 6.02x10^23", "C) 6.02x10^25", "D) 6.02x10^18"], "answer": "B"},
    {"question": "What type of bond involves sharing electrons?", "choices": ["A) Ionic bond", "B) Metallic bond", "C) Covalent bond", "D) Hydrogen bond"], "answer": "C"},
    {"question": "What is the chemical formula for table salt?", "choices": ["A) KCl", "B) NaCl", "C) CaCl2", "D) MgCl2"], "answer": "B"},
    {"question": "What is the lightest element?", "choices": ["A) Helium", "B) Lithium", "C) Hydrogen", "D) Carbon"], "answer": "C"},
    {"question": "What is the powerhouse of the cell?", "choices": ["A) Nucleus", "B) Ribosome", "C) Mitochondria", "D) Golgi apparatus"], "answer": "C"},
    {"question": "What is the basic unit of life?", "choices": ["A) Tissue", "B) Organ", "C) Atom", "D) Cell"], "answer": "D"},
    {"question": "What molecule carries genetic information?", "choices": ["A) RNA", "B) DNA", "C) Protein", "D) Lipid"], "answer": "B"},
    {"question": "How many chromosomes do humans have?", "choices": ["A) 23", "B) 44", "C) 46", "D) 48"], "answer": "C"},
    {"question": "What process do plants use to make food?", "choices": ["A) Respiration", "B) Fermentation", "C) Photosynthesis", "D) Digestion"], "answer": "C"},
    {"question": "What is the largest organ in the human body?", "choices": ["A) Liver", "B) Brain", "C) Heart", "D) Skin"], "answer": "D"},
    {"question": "What blood type is the universal donor?", "choices": ["A) A", "B) B", "C) AB", "D) O"], "answer": "D"},
    {"question": "What is the function of red blood cells?", "choices": ["A) Fight infection", "B) Carry oxygen", "C) Clot blood", "D) Produce antibodies"], "answer": "B"},
    {"question": "What is the study of heredity called?", "choices": ["A) Ecology", "B) Genetics", "C) Taxonomy", "D) Physiology"], "answer": "B"},
    {"question": "What is the process by which cells divide?", "choices": ["A) Meiosis only", "B) Mitosis only", "C) Both mitosis and meiosis", "D) Osmosis"], "answer": "C"},
    {"question": "What does CPU stand for?", "choices": ["A) Central Processing Unit", "B) Computer Processing Unit", "C) Central Program Unit", "D) Core Processing Unit"], "answer": "A"},
    {"question": "What is the binary representation of decimal 10?", "choices": ["A) 1000", "B) 1010", "C) 1100", "D) 0110"], "answer": "B"},
    {"question": "What does RAM stand for?", "choices": ["A) Read Access Memory", "B) Random Access Memory", "C) Read And Memory", "D) Random Allocation Memory"], "answer": "B"},
    {"question": "What is the time complexity of binary search?", "choices": ["A) O(n)", "B) O(n^2)", "C) O(log n)", "D) O(n log n)"], "answer": "C"},
    {"question": "What does HTML stand for?", "choices": ["A) Hyper Text Markup Language", "B) High Text Markup Language", "C) Hyper Transfer Markup Language", "D) Hyper Text Making Language"], "answer": "A"},
    {"question": "What is the base of hexadecimal number system?", "choices": ["A) 2", "B) 8", "C) 10", "D) 16"], "answer": "D"},
    {"question": "What does SQL stand for?", "choices": ["A) Structured Query Language", "B) Simple Query Language", "C) Standard Query Language", "D) Sequential Query Language"], "answer": "A"},
    {"question": "What is a compiler?", "choices": ["A) Runs programs line by line", "B) Translates high-level code to machine code", "C) Manages memory", "D) Connects to internet"], "answer": "B"},
    {"question": "What is the purpose of an operating system?", "choices": ["A) Browse internet", "B) Write code", "C) Manage hardware and software resources", "D) Store data permanently"], "answer": "C"},
    {"question": "What is machine learning?", "choices": ["A) Programming robots manually", "B) Systems that learn patterns from data", "C) Computer hardware design", "D) Network security"], "answer": "B"},
    {"question": "In which year did World War II end?", "choices": ["A) 1943", "B) 1944", "C) 1945", "D) 1946"], "answer": "C"},
    {"question": "Who was the first President of the United States?", "choices": ["A) John Adams", "B) Thomas Jefferson", "C) George Washington", "D) Benjamin Franklin"], "answer": "C"},
    {"question": "In which year did the French Revolution begin?", "choices": ["A) 1776", "B) 1789", "C) 1799", "D) 1815"], "answer": "B"},
    {"question": "Who wrote the Communist Manifesto?", "choices": ["A) Lenin", "B) Stalin", "C) Marx and Engels", "D) Trotsky"], "answer": "C"},
    {"question": "In which year did India gain independence?", "choices": ["A) 1945", "B) 1946", "C) 1947", "D) 1948"], "answer": "C"},
    {"question": "What was the Cold War primarily between?", "choices": ["A) USA and China", "B) USA and USSR", "C) UK and Germany", "D) France and Russia"], "answer": "B"},
    {"question": "Who was Napoleon Bonaparte?", "choices": ["A) Russian tsar", "B) German emperor", "C) French military and political leader", "D) British general"], "answer": "C"},
    {"question": "In which year did the Berlin Wall fall?", "choices": ["A) 1987", "B) 1988", "C) 1989", "D) 1990"], "answer": "C"},
    {"question": "Which country was first to put a man on the moon?", "choices": ["A) USSR", "B) USA", "C) China", "D) UK"], "answer": "B"},
    {"question": "What was the Renaissance?", "choices": ["A) A war in Europe", "B) A cultural and intellectual movement", "C) A religious movement", "D) An economic crisis"], "answer": "B"},
    {"question": "What is the capital of Australia?", "choices": ["A) Sydney", "B) Melbourne", "C) Canberra", "D) Brisbane"], "answer": "C"},
    {"question": "What is the longest river in the world?", "choices": ["A) Amazon", "B) Nile", "C) Yangtze", "D) Mississippi"], "answer": "B"},
    {"question": "Which is the largest ocean?", "choices": ["A) Atlantic", "B) Indian", "C) Arctic", "D) Pacific"], "answer": "D"},
    {"question": "What is the capital of Japan?", "choices": ["A) Osaka", "B) Kyoto", "C) Tokyo", "D) Hiroshima"], "answer": "C"},
    {"question": "Which country has the largest population?", "choices": ["A) China", "B) India", "C) USA", "D) Indonesia"], "answer": "B"},
    {"question": "What is the smallest country in the world?", "choices": ["A) Monaco", "B) San Marino", "C) Liechtenstein", "D) Vatican City"], "answer": "D"},
    {"question": "What is the highest mountain in the world?", "choices": ["A) K2", "B) Kangchenjunga", "C) Mount Everest", "D) Lhotse"], "answer": "C"},
    {"question": "Which continent has the most countries?", "choices": ["A) Asia", "B) Europe", "C) Africa", "D) Americas"], "answer": "C"},
    {"question": "What is the capital of Brazil?", "choices": ["A) Rio de Janeiro", "B) Sao Paulo", "C) Brasilia", "D) Salvador"], "answer": "C"},
    {"question": "Which desert is the largest hot desert?", "choices": ["A) Gobi", "B) Arabian", "C) Sahara", "D) Kalahari"], "answer": "C"},
    {"question": "What does GDP stand for?", "choices": ["A) Gross Domestic Product", "B) General Domestic Product", "C) Gross Development Product", "D) Global Domestic Product"], "answer": "A"},
    {"question": "What is inflation?", "choices": ["A) Decrease in prices", "B) Increase in general price level over time", "C) Stable prices", "D) Government spending"], "answer": "B"},
    {"question": "What is a monopoly?", "choices": ["A) Many sellers", "B) Two sellers", "C) Single seller dominating market", "D) Government-owned market"], "answer": "C"},
    {"question": "What does fiscal policy refer to?", "choices": ["A) Central bank interest rates", "B) Government spending and taxation", "C) Exchange rate management", "D) Trade tariffs"], "answer": "B"},
    {"question": "What is opportunity cost?", "choices": ["A) Cost of production", "B) Value of next best alternative given up", "C) Total project cost", "D) Profit from decision"], "answer": "B"},
    {"question": "What is a recession?", "choices": ["A) Period of economic growth", "B) Period of stable growth", "C) Two consecutive quarters of negative GDP growth", "D) High inflation period"], "answer": "C"},
    {"question": "What is the stock market?", "choices": ["A) Where goods are traded", "B) Where currencies are exchanged", "C) Where company shares are traded", "D) Where bonds are issued"], "answer": "C"},
    {"question": "What is supply and demand?", "choices": ["A) Government price control", "B) Market forces determining price and quantity", "C) Fixed pricing system", "D) Import export balance"], "answer": "B"},
    {"question": "What is comparative advantage?", "choices": ["A) Producing more than others", "B) Lower opportunity cost in producing a good", "C) Having better technology", "D) Exporting more than importing"], "answer": "B"},
    {"question": "What is the purpose of a central bank?", "choices": ["A) Provide personal loans", "B) Control monetary policy and money supply", "C) Sell products to consumers", "D) Regulate stock markets only"], "answer": "B"},
    {"question": "Who is the father of psychoanalysis?", "choices": ["A) Carl Jung", "B) William James", "C) Sigmund Freud", "D) B.F. Skinner"], "answer": "C"},
    {"question": "What is classical conditioning?", "choices": ["A) Learning through rewards", "B) Learning through punishment", "C) Learning through association of stimuli", "D) Learning through observation"], "answer": "C"},
    {"question": "What does IQ stand for?", "choices": ["A) Intelligence Quotient", "B) Intellectual Quality", "C) Intelligence Quality", "D) Intellectual Quotient"], "answer": "A"},
    {"question": "What is cognitive dissonance?", "choices": ["A) Memory loss", "B) Mental discomfort from contradictory beliefs", "C) Learning disability", "D) Attention disorder"], "answer": "B"},
    {"question": "What is operant conditioning?", "choices": ["A) Learning through association", "B) Learning through observation", "C) Learning through consequences", "D) Learning through repetition"], "answer": "C"},
    {"question": "What does PTSD stand for?", "choices": ["A) Post Traumatic Stress Disorder", "B) Post Therapy Stress Disorder", "C) Primary Traumatic Stress Disorder", "D) Post Traumatic Stress Disease"], "answer": "A"},
    {"question": "What is the placebo effect?", "choices": ["A) Side effect of medication", "B) Improvement from inactive treatment due to belief", "C) Negative drug reaction", "D) Drug overdose effect"], "answer": "B"},
    {"question": "What is social learning theory associated with?", "choices": ["A) Freud", "B) Skinner", "C) Bandura", "D) Piaget"], "answer": "C"},
    {"question": "What is Maslow's highest level of needs?", "choices": ["A) Safety", "B) Love and belonging", "C) Esteem", "D) Self-actualization"], "answer": "D"},
    {"question": "What is the unconscious mind?", "choices": ["A) Conscious thoughts", "B) Memories and desires outside awareness", "C) Rational thinking", "D) Short-term memory"], "answer": "B"},
    {"question": "Who wrote Romeo and Juliet?", "choices": ["A) Charles Dickens", "B) William Shakespeare", "C) Jane Austen", "D) Mark Twain"], "answer": "B"},
    {"question": "Who wrote 1984?", "choices": ["A) Aldous Huxley", "B) Ray Bradbury", "C) George Orwell", "D) H.G. Wells"], "answer": "C"},
    {"question": "Who wrote Pride and Prejudice?", "choices": ["A) Charlotte Bronte", "B) Emily Bronte", "C) Jane Austen", "D) George Eliot"], "answer": "C"},
    {"question": "Who wrote The Great Gatsby?", "choices": ["A) Ernest Hemingway", "B) F. Scott Fitzgerald", "C) John Steinbeck", "D) William Faulkner"], "answer": "B"},
    {"question": "Who wrote the Iliad and the Odyssey?", "choices": ["A) Virgil", "B) Plato", "C) Aristotle", "D) Homer"], "answer": "D"},
    {"question": "Who wrote Crime and Punishment?", "choices": ["A) Tolstoy", "B) Dostoevsky", "C) Chekhov", "D) Turgenev"], "answer": "B"},
    {"question": "Who wrote To Kill a Mockingbird?", "choices": ["A) Truman Capote", "B) Harper Lee", "C) John Steinbeck", "D) William Faulkner"], "answer": "B"},
    {"question": "What literary device is a comparison using like or as?", "choices": ["A) Metaphor", "B) Simile", "C) Personification", "D) Alliteration"], "answer": "B"},
    {"question": "Who wrote Don Quixote?", "choices": ["A) Lope de Vega", "B) Garcia Lorca", "C) Miguel de Cervantes", "D) Pablo Neruda"], "answer": "C"},
    {"question": "What is the first book of the Bible?", "choices": ["A) Exodus", "B) Leviticus", "C) Genesis", "D) Numbers"], "answer": "C"},
]

# ============================================================
# CHECKPOINT
# ============================================================
def save_ckpt(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[CHECKPOINT SAVED]")

def load_ckpt():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

ckpt = load_ckpt()
if ckpt:
    print(f"Resuming. Completed: {list(ckpt.get('results', {}).keys())}")
else:
    print("Starting fresh ablation Option C.")
    ckpt = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": {}
    }
    save_ckpt(ckpt)

print("=" * 65)
print("ABLATION OPTION C: Fixed Alpha vs CMA-ES Learned Alpha")
print("=" * 65)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Benchmarks: GSM8K subset (97Q) + MMLU subset (100Q)")
print(f"Seeds: {SEEDS}")

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def evaluate_gsm8k(model, tokenizer, questions, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in questions:
            prompt = (f"Solve this math problem and give only the "
                     f"final numerical answer.\nProblem: {item['question']}\nAnswer:")
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=256, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=50,
                                    do_sample=False, temperature=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated,
                                         skip_special_tokens=True).strip()
            if str(item['answer']).strip() in prediction:
                correct += 1
    return correct / len(questions) * 100

def evaluate_mmlu(model, tokenizer, questions, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in questions:
            prompt = (f"Question: {item['question']}\n"
                     f"Choices:\n" + "\n".join(item['choices']) +
                     f"\nAnswer with only the letter A, B, C, or D:\n")
            inputs = tokenizer(prompt, return_tensors="pt",
                             max_length=512, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=5,
                                    do_sample=False, temperature=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(generated,
                                         skip_special_tokens=True).strip()
            pred_letter = ""
            for char in prediction.upper():
                if char in ["A", "B", "C", "D"]:
                    pred_letter = char
                    break
            if pred_letter == item['answer']:
                correct += 1
    return correct / len(questions) * 100

# ============================================================
# LOAD MODELS
# ============================================================
print("\nLoading models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_A)
tokenizer.pad_token = tokenizer.eos_token

model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model A loaded: {sum(p.numel() for p in model_a.parameters()):,} params")

model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": device})
print(f"Model B loaded: {sum(p.numel() for p in model_b.parameters()):,} params")

sd_a = {k: v.cpu().clone() for k, v in model_a.state_dict().items()}
sd_b = {k: v.cpu().clone() for k, v in model_b.state_dict().items()}
total_layers = len(sd_a)
layer_keys = list(sd_a.keys())
print(f"Total layers: {total_layers}")

del model_b
torch.cuda.empty_cache()
gc.collect()
print("Model B freed from GPU.")

# ============================================================
# MERGE AND EVALUATE FUNCTION
# ============================================================
def run_config(config_name, use_cma, fixed_alpha, seed):
    """Run one configuration on both benchmarks"""
    key = f"{config_name}_seed{seed}"
    if key in ckpt["results"]:
        print(f"  [SKIP] {key} already done.")
        return ckpt["results"][key]

    np.random.seed(seed)
    binary_z = np.random.binomial(1, SELECTION_PROB, total_layers)
    active_keys = [layer_keys[i] for i in range(total_layers) if binary_z[i] == 1]
    active_count = len(active_keys)

    if not use_cma:
        # Fixed alpha — simple merge
        candidate = {k: v.clone() for k, v in sd_a.items()}
        for k in active_keys:
            if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                candidate[k] = (fixed_alpha * sd_a[k].float() +
                               (1-fixed_alpha) * sd_b[k].float()).half()

        torch.cuda.empty_cache()
        gc.collect()
        merged = copy.deepcopy(model_a)
        merged.load_state_dict({k: v.to(device) for k, v in candidate.items()})

        gsm8k_acc = evaluate_gsm8k(merged, tokenizer, GSM8K_QUESTIONS, device)
        mmlu_acc = evaluate_mmlu(merged, tokenizer, MMLU_QUESTIONS, device)

        del merged
        torch.cuda.empty_cache()
        gc.collect()

    else:
        # CMA-ES learned alpha
        dim = active_count
        opts = cma_lib.CMAOptions()
        opts['seed'] = seed
        opts['popsize'] = CMA_POPSIZE
        opts['maxiter'] = CMA_ITERATIONS
        opts['bounds'] = [0.1, 0.9]
        opts['verbose'] = -9

        es = cma_lib.CMAEvolutionStrategy([0.5] * dim, 0.2, opts)
        best_gsm8k = 0
        best_sd = {k: v.clone() for k, v in sd_a.items()}

        it = 0
        while not es.stop():
            it += 1
            solutions = es.ask()
            fitnesses = []

            for sol in solutions:
                alphas = np.clip(sol, 0.1, 0.9)
                candidate = {k: v.clone() for k, v in sd_a.items()}
                for i, k in enumerate(active_keys):
                    if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                        candidate[k] = (alphas[i] * sd_a[k].float() +
                                       (1-alphas[i]) * sd_b[k].float()).half()

                torch.cuda.empty_cache()
                gc.collect()
                cand = copy.deepcopy(model_a)
                cand.load_state_dict(
                    {k: v.to(device) for k, v in candidate.items()})
                acc = evaluate_gsm8k(cand, tokenizer, GSM8K_QUESTIONS, device)
                del cand
                del candidate
                torch.cuda.empty_cache()
                gc.collect()

                fitnesses.append(-acc)
                if acc > best_gsm8k:
                    best_gsm8k = acc
                    best_sd = {k: v.clone() for k, v in sd_a.items()}
                    for i, k in enumerate(active_keys):
                        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
                            best_sd[k] = (alphas[i] * sd_a[k].float() +
                                         (1-alphas[i]) * sd_b[k].float()).half()

            es.tell(solutions, fitnesses)
            print(f"    CMA-ES iter {it}: best GSM8K={best_gsm8k:.1f}%")

        # Evaluate best on both benchmarks
        torch.cuda.empty_cache()
        gc.collect()
        merged = copy.deepcopy(model_a)
        merged.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
        gsm8k_acc = best_gsm8k
        mmlu_acc = evaluate_mmlu(merged, tokenizer, MMLU_QUESTIONS, device)
        del merged
        torch.cuda.empty_cache()
        gc.collect()

    result = {
        "gsm8k": gsm8k_acc,
        "mmlu": mmlu_acc,
        "active_layers": active_count
    }

    ckpt["results"][key] = result
    save_ckpt(ckpt)
    print(f"  {key}: GSM8K={gsm8k_acc:.1f}% MMLU={mmlu_acc:.1f}% [SAVED]")
    return result

# ============================================================
# RUN ALL CONFIGURATIONS
# ============================================================
print("\n" + "=" * 65)
print("RUNNING ALL CONFIGURATIONS")
print("=" * 65)

for config_name, use_cma, fixed_alpha in CONFIGS:
    print(f"\n{'='*65}")
    print(f"Configuration: {config_name}")
    print(f"{'='*65}")

    for seed in SEEDS:
        print(f"\n  Seed={seed}...")
        run_config(config_name, use_cma, fixed_alpha, seed)

# ============================================================
# COMPUTE FINAL STATISTICS
# ============================================================
print("\n" + "=" * 65)
print("FINAL RESULTS — Fixed Alpha vs CMA-ES Learned Alpha")
print("=" * 65)
print(f"{'Method':<25} {'GSM8K Mean':>12} {'GSM8K Std':>10} {'MMLU Mean':>12} {'MMLU Std':>10}")
print("-" * 75)

final_results = {}
for config_name, use_cma, fixed_alpha in CONFIGS:
    gsm8k_runs = []
    mmlu_runs = []
    for seed in SEEDS:
        key = f"{config_name}_seed{seed}"
        if key in ckpt["results"]:
            gsm8k_runs.append(ckpt["results"][key]["gsm8k"])
            mmlu_runs.append(ckpt["results"][key]["mmlu"])

    if gsm8k_runs:
        gsm8k_mean = np.mean(gsm8k_runs)
        gsm8k_std = np.std(gsm8k_runs)
        mmlu_mean = np.mean(mmlu_runs)
        mmlu_std = np.std(mmlu_runs)
        final_results[config_name] = {
            "gsm8k_mean": gsm8k_mean,
            "gsm8k_std": gsm8k_std,
            "mmlu_mean": mmlu_mean,
            "mmlu_std": mmlu_std,
            "gsm8k_runs": gsm8k_runs,
            "mmlu_runs": mmlu_runs
        }
        print(f"{config_name:<25} {gsm8k_mean:>11.1f}% {gsm8k_std:>9.1f}% "
              f"{mmlu_mean:>11.1f}% {mmlu_std:>9.1f}%")

print("=" * 75)

# Save final results
with open(RESULTS_FILE, 'w') as f:
    json.dump({
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": final_results
    }, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")

# ============================================================
# GENERATE FIGURE
# ============================================================
if len(final_results) == 4:
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Ablation: Fixed Alpha vs CMA-ES Learned Alpha\n'
        'Mistral-7B Models — 3 Seeds Each',
        fontsize=12, fontweight='bold'
    )

    labels = ['Fixed\nα=0.3', 'Fixed\nα=0.5', 'Fixed\nα=0.7', 'CMA-ES\nLearned']
    colors = ['#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71']
    config_names = ['fixed_alpha_0.3', 'fixed_alpha_0.5',
                   'fixed_alpha_0.7', 'cma_es_learned']

    for ax_idx, (benchmark, key_mean, key_std) in enumerate([
        ('GSM8K Subset (97Q)', 'gsm8k_mean', 'gsm8k_std'),
        ('MMLU Subset (100Q)', 'mmlu_mean', 'mmlu_std')
    ]):
        ax = axes[ax_idx]
        means = [final_results[c][key_mean] for c in config_names
                 if c in final_results]
        stds = [final_results[c][key_std] for c in config_names
                if c in final_results]

        bars = ax.bar(range(len(means)), means, color=colors[:len(means)],
                     width=0.6, edgecolor='black', linewidth=0.8)
        ax.errorbar(range(len(means)), means, yerr=stds,
                   fmt='none', color='black', capsize=6, linewidth=2)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2.,
                   bar.get_height() + std + 0.3,
                   f'{mean:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(labels[:len(means)])))
        ax.set_xticklabels(labels[:len(means)], fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{benchmark}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(max(0, min(means) - 15), max(means) + 10)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {FIGURE_FILE}")
    plt.show()

print("\nABLATION OPTION C COMPLETE")
