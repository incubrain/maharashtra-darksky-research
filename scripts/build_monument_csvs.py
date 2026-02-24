#!/usr/bin/env python3
"""
Build per-district monument CSV files from the raw monument data.

Source: Maharashtra State Archaeology Department official monument lists
(Ratnagiri, Nashik, Pune, Aurangabad, Nanded, Nagpur divisions).

Output: data/monuments/<district>.csv for each district.

Usage:
    python scripts/build_monument_csvs.py
"""

import csv
import os
import re

OUTPUT_DIR = "data/monuments"

# All 385 monuments parsed from the official lists.
# Format: (name, monument_type, place, taluka, district)
# notification_status omitted from inline data; added as column.

MONUMENTS = [
    # ═══ RATNAGIRI DIVISION ═══
    # District Palghar
    ("Shirgaon Fort", "Fort", "Shirgaon", "Palghar", "Palghar", "Final"),
    ("Pangadhi Fort", "Fort", "Mauje-Kelve", "Palghar", "Palghar", "First"),
    # District Thane
    ("Khandeshwari Caves", "Caves", "Lonad", "Thane", "Thane", "Final"),
    ("Ghodbandar Fort", "Fort", "Ghodbandar", "Thane", "Thane", "Final"),
    # District Mumbai
    ("August Kranti Maidan", "Other", "Grant Road", "Mumbai", "Mumbai", "Final"),
    ("Gateway of India", "Other", "Apollo Bundar", "Mumbai", "Mumbai", "Final"),
    ("Dean Bunglow (Rudyard Kipling)", "Other", "Mumbai", "Mumbai", "Mumbai", "Final"),
    ("Dharavi Fort", "Fort", "Dharavi", "Mumbai", "Mumbai", "Final"),
    ("Bandra Fort", "Fort", "Bandra", "Mumbai", "Mumbai", "Final"),
    ("Banganga Talav", "Other", "Walkeshwar", "Mumbai", "Mumbai", "Final"),
    ("Mahim Fort", "Fort", "Mahim", "Mumbai Suburbs", "Mumbai", "Final"),
    ("Worli Fort", "Fort", "Worli", "Mumbai Suburbs", "Mumbai", "Final"),
    ("Shewri Fort", "Fort", "Shewri", "Mumbai", "Mumbai", "Final"),
    ("Saint George Fort", "Fort", "Mumbai", "Mumbai", "Mumbai", "Final"),
    # District Raigad
    ("Underi Fort", "Fort", "Underi", "Alibaug", "Raigad", "Final"),
    ("Mangad Fort", "Fort", "Mashidwadi", "Mangaon", "Raigad", "Final"),
    ("Vasudeo Balwant Phadke Birthplace", "Other", "Shirdhon", "Panvel", "Raigad", "Final"),
    ("Sarkhel Kanhoji Angre Samadhi", "Other", "Alibaug", "Alibaug", "Raigad", "Final"),
    ("Karnala Fort", "Fort", "Kalhe", "Panvel", "Raigad", "First"),
    ("Khanderi Fort", "Fort", "Khanderi", "Alibaug", "Raigad", "First"),
    # District Ratnagiri
    ("Karneshwar Temple", "Temple", "Kasba Sangameshwar", "Ratnagiri", "Ratnagiri", "Final"),
    ("Gopalgad Fort", "Fort", "Anganvel", "Guhagar", "Ratnagiri", "Final"),
    ("Goa Fort", "Fort", "Harne", "Dapoli", "Ratnagiri", "Final"),
    ("Thiba Palace", "Other", "Nachne", "Ratnagiri", "Ratnagiri", "Final"),
    ("Thiba King and Queen Tomb", "Other", "Nachne", "Ratnagiri", "Ratnagiri", "Final"),
    ("Purnagad", "Fort", "Purnagad", "Ratnagiri", "Ratnagiri", "Final"),
    ("Buddhist Caves", "Caves", "Khed", "Khed", "Ratnagiri", "Final"),
    ("Bankot Fort", "Fort", "Mandangad", "Ratnagiri", "Ratnagiri", "Final"),
    ("Mahipatgad Fort", "Fort", "Nigudwadi", "Sangameshwar", "Ratnagiri", "Final"),
    ("Yashwantgad (Nate)", "Fort", "Nate", "Rajapur", "Ratnagiri", "Final"),
    ("Rasalgad", "Fort", "Rasalgad", "Khed", "Ratnagiri", "Final"),
    ("Lokmanya Tilak Birthplace", "Other", "Ratnagiri", "Ratnagiri", "Ratnagiri", "Final"),
    ("Petroglyph at Devihasol", "Archaeological Site", "Rajapur", "Rajapur", "Ratnagiri", "First"),
    ("Petroglyph at Bhagvatinagar", "Archaeological Site", "Ratnagiri", "Ratnagiri", "Ratnagiri", "First"),
    ("Petroglyph at Chave", "Archaeological Site", "Ratnagiri", "Ratnagiri", "Ratnagiri", "First"),
    ("Petroglyph at Wadirundhe", "Archaeological Site", "Rajapur", "Rajapur", "Ratnagiri", "First"),
    ("Petroglyph at Kapadgaon", "Archaeological Site", "Ratnagiri", "Ratnagiri", "Ratnagiri", "First"),
    ("Petroglyph at Barsu No. 2", "Archaeological Site", "Rajapur", "Rajapur", "Ratnagiri", "First"),
    ("Petroglyph at Deud", "Archaeological Site", "Ratnagiri", "Ratnagiri", "Ratnagiri", "First"),
    ("Petroglyph at Rajapur", "Archaeological Site", "Rajapur", "Rajapur", "Ratnagiri", "First"),
    ("Petroglyph at Ukshi", "Archaeological Site", "Ratnagiri", "Ratnagiri", "Ratnagiri", "First"),
    # District Sindhudurg
    ("Dutch Factory", "Other", "Vengurla", "Vengurla", "Sindhudurg", "Final"),
    ("Bharatgad", "Fort", "Masure", "Malvan", "Sindhudurg", "Final"),
    ("Yashwantgad (Sukalbhat)", "Fort", "Sukalbhat", "Vengurla", "Sindhudurg", "Final"),

    # ═══ NASHIK DIVISION ═══
    # District Nandurbar
    ("Jain Cave", "Caves", "MahodaTarfeHaveli", "Shahada", "Nandurbar", "Final"),
    # District Dhule
    ("Kalikadevi Temple", "Temple", "Shirud", "Dhule", "Dhule", "Final"),
    ("Jain Temple (Nizampur)", "Temple", "Nizampur", "Sakri", "Dhule", "First"),
    ("Harba Temple / Mahadev Temple", "Temple", "Methi", "Sindhkhed", "Dhule", "First"),
    ("Bhavani Temple / Vishnu Temple (Balaji)", "Temple", "Methi", "Sindhkhed", "Dhule", "First"),
    ("Laling Fort", "Fort", "Laling", "Dhule", "Dhule", "First"),
    # District Nashik
    ("Ankai Fort", "Fort", "Ankai", "Yeola", "Nashik", "Final"),
    ("Tankai Fort", "Fort", "Ankai", "Yeola", "Nashik", "Final"),
    ("Arnath Jain Caves", "Caves", "Anjaneri", "Trimbak", "Nashik", "First"),
    ("Indraleshwar Temple", "Temple", "Trimbakeshwar", "Nashik City", "Nashik", "First"),
    ("Kushavart Tirth", "Other", "Trimbakeshwar", "Nashik", "Nashik", "First"),
    ("Galna Fort", "Fort", "Galna", "Malegaon", "Nashik", "Final"),
    ("Tribhuvaneshwar Temple", "Temple", "Trimbakeshwar", "Nashik", "Nashik", "First"),
    ("Tatoba Temple", "Temple", "Odha", "Nashik", "Nashik", "Final"),
    ("Nilkantheswar Mahadev Temple (Nashik)", "Temple", "Nashik", "Nashik", "Nashik", "First"),
    ("Parshvanath Jain Caves", "Caves", "Anjaneri", "Trimbak", "Nashik", "First"),
    ("Ballaleshwar Temple", "Temple", "Trimbak", "Trimbak", "Nashik", "Final"),
    ("Malegaon Fort", "Fort", "Malegaon", "Malegaon", "Nashik", "Final"),
    ("Mahadev Temple (Devlana)", "Temple", "Devlana", "Satana", "Nashik", "First"),
    ("Mulher Fort", "Fort", "Mulher", "Satana", "Nashik", "First"),
    ("Rangmahal (Chandwad)", "Other", "Chandwad", "Chandwad", "Nashik", "Final"),
    ("Raghveshwar Temple", "Temple", "Chichondi", "Yeola", "Nashik", "First"),
    ("Renuka Devi Temple", "Temple", "Chandwad", "Chandwad", "Nashik", "Final"),
    ("Vishnu Temple (Dhodambe)", "Temple", "Dhodambe", "Chandwad", "Nashik", "Final"),
    ("Vaijeshwar Temple", "Temple", "Vavi", "Sinnar", "Nashik", "Final"),
    ("Vateshwar Mahadev Temple", "Temple", "Dhodambe", "Chandwad", "Nashik", "First"),
    ("Sundar Narayan Temple", "Temple", "Nashik", "Nashik", "Nashik", "Final"),
    ("Sarkawada", "Other", "Nashik", "Nashik", "Nashik", "Final"),
    ("Salher Fort", "Fort", "Salher", "Satana", "Nashik", "First"),
    ("Swatantraveer Savarkar Birthplace", "Other", "Bhagur", "Nashik", "Nashik", "Final"),
    ("Hatgad Fort", "Fort", "Hatgad", "Surgana", "Nashik", "First"),
    ("Mahadev Temple (Deoli-Karad)", "Temple", "Deoli-Karad", "Kalvan", "Nashik", "First"),
    # District Jalgaon
    ("Kapileshwar Temple", "Temple", "Velhale", "Jalgaon", "Jalgaon", "First"),
    ("Jhulte Manore", "Other", "Farkanda", "Erandol", "Jalgaon", "Final"),
    ("Pandavwada Masjid", "Other", "Erandol", "Erandol", "Jalgaon", "Final"),
    ("Parola Fort", "Fort", "Parola", "Parola", "Jalgaon", "Final"),
    ("Ves (Amalner Fort Gate and Bastion)", "Other", "Amalner", "Amalner", "Jalgaon", "Final"),
    ("Shiv Temple (Tondapur)", "Temple", "Tondapur", "Jamner", "Jalgaon", "Final"),
    # District Ahmednagar
    ("Kharda Fort", "Fort", "Kharda", "Jamkhed", "Ahmednagar", "First"),
    ("Gadhi - Ahilyabai Holkar Birthplace", "Temple", "Chondi", "Jamkhed", "Ahmednagar", "Final"),
    ("Raghobadada Wada", "Other", "Kopargaon", "Kopargaon", "Ahmednagar", "Final"),
    ("Lakshmi Temple (Shrigonda)", "Temple", "Shrigonda", "Shrigonda", "Ahmednagar", "Final"),
    ("Raghaveshwar Mahadev Temple", "Temple", "Kopargaon", "Kopargaon", "Ahmednagar", "Final"),
    ("Siddheshwar Mahadev Temple (Limpangaon)", "Temple", "Limpangaon", "Shrigonda", "Ahmednagar", "Final"),
    ("Senapati Bapat Birthplace", "Other", "Parner", "Parner", "Ahmednagar", "Final"),
    ("Harinarayan Math", "Other", "Benwadi", "Karjat", "Ahmednagar", "Final"),
    ("Nimbalkar Gadhi, Chatri, Kareshwar Temple", "Other", "Kharda", "Jamkhed", "Ahmednagar", "Final"),

    # ═══ PUNE DIVISION ═══
    # District Pune
    ("Koyrigad Fort", "Fort", "Ambavane", "Mulshi", "Pune", "Final"),
    ("Kukdeshwar Temple", "Temple", "Pur", "Junnar", "Pune", "Final"),
    ("Khandoba Temple (Jejuri)", "Temple", "Jejuri", "Baramati", "Pune", "Final"),
    ("Torna Fort", "Fort", "Velhe", "Velhe", "Pune", "Final"),
    ("Tanaji Malusare Samadhi", "Other", "Sinhagad", "Haveli", "Pune", "Final"),
    ("Chalcolithic Settlement (Inamgaon)", "Archaeological Site", "Inamgaon", "Shirur", "Pune", "Final"),
    ("Nageshwar Temple (Pune City)", "Temple", "Pune City", "Haveli", "Pune", "Final"),
    ("Narsinha Temple (Nira)", "Temple", "Nira", "Indapur", "Pune", "Final"),
    ("Bhandara Buddhist Cave Complex", "Caves", "Bhandara Hill", "Maval", "Pune", "First"),
    ("Mastani Tomb", "Other", "Pabal", "Shirur", "Pune", "Final"),
    ("Mahadev Temple (Tulapur)", "Temple", "Tulapur", "Haveli", "Pune", "Final"),
    ("Mahatma Phule Wada", "Other", "Pune", "Haveli", "Pune", "Final"),
    ("Mahadev Temple and Datta Temple Complex", "Temple", "Loni Bhapkar", "Baramati", "Pune", "First"),
    ("Rajgad Fort", "Fort", "Gunjavane", "Velhe", "Pune", "Final"),
    ("Raireshwar Temple", "Temple", "Rairi", "Bhor", "Pune", "First"),
    ("Vishrambag Wada", "Other", "Pune", "Haveli", "Pune", "Final"),
    ("Sangramdurg Fort (Chakan)", "Fort", "Chakan", "Khed", "Pune", "Final"),
    ("Saint Crispin Church", "Other", "Pune", "Haveli", "Pune", "Final"),
    ("Santaji Jagnade Maharaj Samadhi", "Other", "Subumbare", "Maval", "Pune", "Final"),
    ("Sambhaji Maharaj Samadhi", "Other", "Vadu", "Koregaon", "Pune", "Final"),
    ("Sardar Kanhoji Jedhe Wada", "Other", "Kari", "Bhor", "Pune", "Final"),
    ("Sinhagad Fort", "Fort", "Donja", "Haveli", "Pune", "Final"),
    ("Hutatma Rajguru Wada", "Other", "Rajgurunagar", "Khed", "Pune", "Final"),
    # District Satara
    ("Dr. Babasaheb Ambedkar Birthplace (Satara)", "Other", "Satara", "Satara", "Satara", "Final"),
    ("Bhairavnath Temple", "Temple", "Kinkali", "Wai", "Satara", "Final"),
    ("Vasudeo Swami Math", "Other", "Kaneri", "Khandala", "Satara", "Final"),
    ("Sarsenapati Hambirrao Mohite Samadhi", "Other", "Talbid", "Karhad", "Satara", "Final"),
    ("Savitribai Phule Birthplace", "Other", "Naigaon", "Khandala", "Satara", "Final"),
    # District Solapur
    ("Mahadev Temple (Kasegaon)", "Temple", "Kasegaon", "S. Solapur", "Solapur", "Final"),
    ("Sangameshwar and Murlidhar Temple", "Temple", "Hattarsag Kudal", "S. Solapur", "Solapur", "Final"),
    # District Sangli
    ("Kalammadevi Temple", "Temple", "Mahuli", "Khanapur", "Sangli", "Final"),
    ("Yashwantrao Chavan Birthplace", "Other", "Devrashtre", "Khanapur", "Sangli", "Final"),
    # District Kolhapur
    ("Pandavdara Shaiva Caves", "Caves", "Dalvewadi", "Shahuwadi", "Kolhapur", "Final"),
    ("Bajiprabhu and Phulaji Deshpande Samadhi", "Other", "Bhattali", "Shahuwadi", "Kolhapur", "Final"),
    ("Bhudargad", "Fort", "Peth Shivapur", "Bhudargad", "Kolhapur", "Final"),
    ("Mahadeo Temple (Aare)", "Temple", "Aare", "Karvir", "Kolhapur", "Final"),
    ("Lakshmi Vilas Palace", "Other", "Karvir", "Kolhapur", "Kolhapur", "Final"),
    ("Rangana Fort", "Fort", "Chikewadi", "Bhudargad", "Kolhapur", "Final"),
    ("Ramchandra Amatya Samadhi", "Other", "Panhala", "Panhala", "Kolhapur", "Final"),
    ("Vishalgad Fort", "Fort", "Gajapur", "Shahuwadi", "Kolhapur", "Final"),

    # ═══ AURANGABAD DIVISION ═══
    # District Aurangabad
    ("Ajantha Sarai", "Other", "Ajanta", "Sillod", "Aurangabad", "Final"),
    ("Antur Fort", "Fort", "Antur", "Kannad", "Aurangabad", "Final"),
    ("Azamshah Tomb", "Other", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Abul Hasan Tanashah Tomb", "Other", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Asafjahan Tomb", "Other", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Delhi Gate (Aurangabad)", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "First"),
    ("Narsimha Sculpture (Paithan)", "Other", "Paithan", "Paithan", "Aurangabad", "First"),
    ("Makai Gate", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "First"),
    ("Lala Hardol Samadhi", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "First"),
    ("Sheshashayi Vishnu (Paithan)", "Other", "Paithan", "Paithan", "Aurangabad", "First"),
    ("Kali Masjid", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Khandoba Temple (Satara, Aurangabad)", "Temple", "Satara", "Aurangabad", "Aurangabad", "Final"),
    ("Khan-e-Jahan Baug", "Other", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Ghatotkach Caves", "Caves", "Jinjala", "Soygaon", "Aurangabad", "Final"),
    ("Chowk Masjid", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Jogeshwari Devi Caves", "Caves", "Ghatnandra", "Sillod", "Aurangabad", "Final"),
    ("Jame Masjid - Asafjahan (Ajanta)", "Other", "Ajanta", "Sillod", "Aurangabad", "Final"),
    ("Tirthstambha (Paithan)", "Other", "Paithan", "Paithan", "Aurangabad", "Final"),
    ("Taltam Fort", "Fort", "Jinjala", "Soygaon", "Aurangabad", "Final"),
    ("Nageshwar Temple (Rahimabad)", "Temple", "Rahimabad", "Sillod", "Aurangabad", "Final"),
    ("Navkhanda Palace", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Nasirjung Tomb", "Other", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Neolithic Site (Aurangabad)", "Archaeological Site", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Neolithic Site (Khultabad)", "Archaeological Site", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Panchakki", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Archaeological Site - Yadav Period (Rahimabad)", "Archaeological Site", "Rahimabad", "Sillod", "Aurangabad", "Final"),
    ("Baitulwadi Fort", "Fort", "Baitulwadi", "Soygaon", "Aurangabad", "Final"),
    ("Bani Begum Baug", "Other", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Bhadkal Gate", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Munim Baug", "Other", "Khultabad", "Khultabad", "Aurangabad", "Final"),
    ("Maloji Raje Bhosale Gadhi Remains", "Other", "Verul", "Khultabad", "Aurangabad", "Final"),
    ("Rudreshwar Caves", "Caves", "Baitulwadi", "Soygaon", "Aurangabad", "Final"),
    ("Lal Masjid", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Vadeshwar Mahadev Temple", "Temple", "Ambhai", "Sillod", "Aurangabad", "Final"),
    ("Pillar Inscription (Antur)", "Other", "Antur", "Kannad", "Aurangabad", "Final"),
    ("Shahgunj Masjid", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Shahi Hamam (Daulatabad)", "Other", "Daulatabad", "Aurangabad", "Aurangabad", "Final"),
    ("Soneri Mahal", "Other", "Aurangabad", "Aurangabad", "Aurangabad", "Final"),
    ("Salarjung Baradari", "Other", "Ajanta", "Sillod", "Aurangabad", "Final"),
    ("Sarai (Fardapur)", "Other", "Fardapur", "Soygaon", "Aurangabad", "Final"),
    # District Jalna
    ("Jambuvant Temple", "Temple", "Jamkhed", "Ambad", "Jalna", "Final"),
    ("Prehistoric Site (Ambad)", "Archaeological Site", "Ambad", "Ambad", "Jalna", "Final"),
    ("Prehistoric Site (Jalna)", "Archaeological Site", "Jalna", "Jalna", "Jalna", "Final"),
    ("Prehistoric Site (Bhokardan)", "Archaeological Site", "Bhokardan", "Jalna", "Jalna", "Final"),
    ("Bhokardan Caves", "Caves", "Bhokardan", "Bhokardan", "Jalna", "Final"),
    ("Mahadev Temple (Anva)", "Temple", "Anva", "Bhokardan", "Jalna", "Final"),
    ("Amargad Fort", "Fort", "Mantha", "Mantha", "Jalna", "Final"),
    # District Beed
    ("Amleshwar Temple", "Temple", "Ambejogai", "Ambejogai", "Beed", "First"),
    ("Adya Kavi Shri Mukundraj Samadhi", "Other", "Ambejogai", "Beed", "Beed", "Final"),
    ("Kankaleshwar Temple", "Temple", "Beed", "Beed", "Beed", "Final"),
    ("Kedareshwar Temple (Dharmapuri)", "Temple", "Dharmapuri", "Ambejogai", "Beed", "Final"),
    ("Kotwali Gate (Beed)", "Other", "Beed", "Beed", "Beed", "Final"),
    ("Kholeshwar Temple", "Temple", "Ambejogai", "Ambejogai", "Beed", "First"),
    ("Khandeshwari Temple (Beed)", "Temple", "Beed", "Beed", "Beed", "Final"),
    ("Gunj Gate", "Other", "Beed", "Beed", "Beed", "Final"),
    ("Jogi Sabhamandap", "Caves", "Ambejogai", "Ambejogai", "Beed", "Final"),
    ("Jame Masjid (Beed)", "Other", "Beed", "Beed", "Beed", "Final"),
    ("Dharur Fort", "Fort", "Dharur", "Dharur", "Beed", "Final"),
    ("Dharmapuri Fort", "Fort", "Dharmapuri", "Parli Vaijanath", "Beed", "Final"),
    ("Dhonda Gate", "Other", "Beed", "Beed", "Beed", "Final"),
    ("Pir Balashah Dargah", "Other", "Beed", "Beed", "Beed", "Final"),
    ("Mahadev Temple (Patoda)", "Temple", "Patoda", "Patoda", "Beed", "Final"),
    ("Mohammad Bin Tughlaq Tomb", "Other", "Karjani", "Beed", "Beed", "Final"),
    ("Rajuri Gate", "Other", "Beed", "Beed", "Beed", "Final"),
    ("Ran Khamb", "Other", "Beed", "Beed", "Beed", "Final"),
    ("Rani Annapurnabai Samadhi", "Other", "Nandurghat", "Kej", "Beed", "Final"),
    ("Shehenshawali Dargah", "Other", "Beed", "Beed", "Beed", "Final"),
    # District Osmanabad
    ("Uttareshwar Temple", "Temple", "Ter", "Osmanabad", "Osmanabad", "Final"),
    ("Chambhar Caves", "Caves", "Osmanabad", "Osmanabad", "Osmanabad", "Final"),
    ("Trivikram Temple", "Temple", "Ter", "Osmanabad", "Osmanabad", "Final"),
    ("Tirthkund (Ter)", "Other", "Ter", "Osmanabad", "Osmanabad", "Final"),
    ("Archaeological Site - Kot Tekdi", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Kisan Mhaske Survey No. 5", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Godavari Tekdi", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Bairag Pandhar", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Mulani Tekdi", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Mahar Tekdi", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Renuka Tekdi", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Suleman Tekdi", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Archaeological Site - Malak Govind", "Archaeological Site", "Ter", "Osmanabad", "Osmanabad", "First"),
    ("Ghummaz (Sastur)", "Other", "Sastur", "Lohara", "Osmanabad", "First"),
    ("Osmanabad Caves", "Caves", "Osmanabad", "Osmanabad", "Osmanabad", "Final"),
    ("Naldurg Fort", "Fort", "Naldurg", "Tuljapur", "Osmanabad", "Final"),
    ("Prehistoric Stone Circle (Shendri)", "Archaeological Site", "Shendri", "Paranda", "Osmanabad", "Final"),
    ("Paranda Fort", "Fort", "Paranda", "Paranda", "Osmanabad", "Final"),
    ("Tulja Bhawani Temple", "Temple", "Tuljapur", "Tuljapur", "Osmanabad", "Final"),
    ("Mahadev Temple (Umarga)", "Temple", "Umarga", "Umarga", "Osmanabad", "Final"),
    ("Mahadev Temple (Mankeshwar)", "Temple", "Mankeshwar", "Paranda", "Osmanabad", "Final"),
    ("Mahalakshmi Temple (Jagji)", "Temple", "Jagji", "Osmanabad", "Osmanabad", "Final"),
    ("Lavni Ghummaz", "Other", "Tuljapur", "Tuljapur", "Osmanabad", "Final"),
    ("Shivguru Samadhi Temple", "Temple", "Osmanabad", "Osmanabad", "Osmanabad", "Final"),
    ("Sant Goroba Kaka Residence", "Other", "Ter", "Osmanabad", "Osmanabad", "Final"),
    ("Hazrat Shamsuddin Dargah (Osmanabad)", "Other", "Osmanabad", "Osmanabad", "Osmanabad", "Final"),
    ("Hindu Temple and Inscription (Murum)", "Temple", "Murum", "Murum", "Osmanabad", "Final"),

    # ═══ NANDED DIVISION ═══
    # District Parbhani
    ("Ukandeshwar Temple", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Khurachi Aai Temple", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Gokuleshwar Temple", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Ganapati Temple (Charthana)", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Gupteshwar Temple", "Temple", "Dharasur", "Gangakhed", "Parbhani", "Final"),
    ("Rutuvihir Temple", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Jod Mahadev Temple", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Jain Temple (Charthana)", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Jain Temple (Jintur)", "Temple", "Jintur", "Jintur", "Parbhani", "Final"),
    ("Jama Masjid (Parbhani)", "Other", "Parbhani", "Parbhani", "Parbhani", "Final"),
    ("Deepmal (Charthana)", "Other", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Nageshwar Temple (Charthana)", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Narsimha Temple (Charthana)", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Panch Pandav Temple (Erandol)", "Temple", "Erandol", "Parbhani", "Parbhani", "Final"),
    ("Pathari Fort", "Fort", "Pathari", "Pathari", "Parbhani", "Final"),
    ("Mahadev Temple (Charthana)", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Mahadev Temple (Arandeshwar)", "Temple", "Arandeshwar", "Parbhani", "Parbhani", "Final"),
    ("Roshankhan Tomb", "Other", "Parbhani", "Parbhani", "Parbhani", "Final"),
    ("Shah Mastan Dargah", "Other", "Jintur", "Jintur", "Parbhani", "Final"),
    ("Shah Shamsuddin Dargah (Jintur)", "Other", "Jintur", "Jintur", "Parbhani", "Final"),
    ("Sutareshwar Temple", "Temple", "Charthana", "Jintur", "Parbhani", "Final"),
    ("Siddheshwar Mahadev Temple (Bhosi)", "Temple", "Bhosi", "Jintur", "Parbhani", "Final"),
    ("Sant Janabai Samadhi", "Other", "Gangakhed", "Gangakhed", "Parbhani", "Final"),
    ("Hazrat Sayed Shah Dome (Konri)", "Other", "Konri", "Jintur", "Parbhani", "Final"),
    ("Hanuman Temple (Borvad)", "Temple", "Borvad", "Jintur", "Parbhani", "Final"),
    ("Hanuman Temple (Charbatula)", "Temple", "Charbatula", "Jintur", "Parbhani", "Final"),
    # District Hingoli
    ("Kazi Sahaeb Masjid", "Other", "Vasmat", "Vasmat", "Hingoli", "Final"),
    ("Khan-e-Alam Dargah", "Other", "Vasmat", "Vasmat", "Hingoli", "Final"),
    ("Jame Masjid and Shah Tankali Shah", "Other", "Aundha", "Aundha", "Hingoli", "Final"),
    ("Nagnath Mahadev Temple", "Temple", "Aundha", "Aundha", "Hingoli", "Final"),
    ("Narsimha Temple (Narsi)", "Temple", "Narsi", "Hingoli", "Hingoli", "Final"),
    ("Panch Pandav Temple - Jain (Aundha)", "Temple", "Aundha", "Aundha", "Hingoli", "Final"),
    ("Fort Ruins (Anthali)", "Fort", "Anthali", "Vasmat", "Hingoli", "Final"),
    ("Temple and Well (Bamani)", "Temple", "Bamani", "Hingoli", "Hingoli", "Final"),
    ("Mahadev Temple (Bamani)", "Temple", "Bamani", "Hingoli", "Hingoli", "Final"),
    ("Vadgaon Fort", "Fort", "Vadgaon", "Kalmanuri", "Hingoli", "Final"),
    ("Hindu Temple (Araldhare)", "Temple", "Araldhare", "Vasmat", "Hingoli", "Final"),
    ("Hemadpanti Temple (Bamani)", "Temple", "Bamani", "Hingoli", "Hingoli", "Final"),
    ("Sant Namdeo Birthplace", "Other", "Narsi", "Hingoli", "Hingoli", "Final"),
    # District Latur
    ("Ausa Fort", "Fort", "Ausa", "Ausa", "Latur", "Final"),
    ("Udgir Fort", "Fort", "Udgir", "Udgir", "Latur", "Final"),
    ("Jame Masjid (Ausa)", "Other", "Ausa", "Ausa", "Latur", "Final"),
    ("Devi Temple (Killari)", "Temple", "Killari", "Ausa", "Latur", "Final"),
    ("Baug-e-Hissam", "Other", "Udgir", "Udgir", "Latur", "Final"),
    # District Nanded
    ("Idgah (Mahur)", "Other", "Mahur", "Mahur", "Nanded", "Final"),
    ("Unkeshwar Temple", "Temple", "Unkeshwar", "Kinwat", "Nanded", "Final"),
    ("Kandhar Fort", "Fort", "Kandhar", "Kandhar", "Nanded", "Final"),
    ("Gurudwara (Nanded)", "Other", "Nanded", "Nanded", "Nanded", "Final"),
    ("Jama Masjid - Malik Amber (Nanded)", "Other", "Nanded", "Nanded", "Nanded", "Final"),
    ("Darbari Masjid - Qutubshahi", "Other", "Nanded", "Nanded", "Nanded", "Final"),
    ("Dasharatheshwar Temple and Well", "Temple", "Mukhed", "Mukhed", "Nanded", "Final"),
    ("Nandagiri Fort", "Fort", "Nanded", "Nanded", "Nanded", "Final"),
    ("Nandi and Temple Remains (Raiwadi)", "Temple", "Raiwadi", "Loha", "Nanded", "Final"),
    ("Narsimha Temple (Shelgaon)", "Temple", "Shelgaon", "Deglur", "Nanded", "Final"),
    ("Nandi Temple and Tank (Hottal)", "Temple", "Hottal", "Deglur", "Nanded", "First"),
    ("Narsimha Temple Complex (Shankhatirth)", "Temple", "Shankhatirth", "Mukhed", "Nanded", "First"),
    ("Parvati Temple (Hottal)", "Temple", "Hottal", "Deglur", "Nanded", "First"),
    ("Mahadev Temple (Hottal)", "Temple", "Hottal", "Deglur", "Nanded", "First"),
    ("Hatikhana (Mahur)", "Other", "Mahur", "Mahur", "Nanded", "First"),
    ("Pandavleni (Mahur)", "Caves", "Mahur", "Mahur", "Nanded", "Final"),
    ("Parmeshwar Temple (Hottal)", "Temple", "Hottal", "Deglur", "Nanded", "Final"),
    ("Parmeshwar Temple (Shirdhon)", "Temple", "Shirdhon", "Kandhar", "Nanded", "Final"),
    ("Bhognarsimha and Ramlinga Temple", "Temple", "Raher", "Naigaon", "Nanded", "Final"),
    ("Bhavani Temple (Pala)", "Temple", "Pala", "Mukhed", "Nanded", "Final"),
    ("Matrutirth Talav (Mahur)", "Other", "Mahur", "Mahur", "Nanded", "Final"),
    ("Mahadev Temple (Yevati)", "Temple", "Yevati", "Dharmabad", "Nanded", "Final"),
    ("Mahadev Temple (Khanapur, Nanded)", "Temple", "Khanapur", "Deglur", "Nanded", "Final"),
    ("Mahadev Temple (Hadgaon)", "Temple", "Hadgaon", "Hadgaon", "Nanded", "Final"),
    ("Mahur Fort", "Fort", "Mahur", "Mahur", "Nanded", "Final"),
    ("Yognarsimha Temple (Raher)", "Temple", "Raher", "Naigaon", "Nanded", "Final"),
    ("Renukadevi Temple (Mahur)", "Temple", "Mahur", "Mahur", "Nanded", "Final"),
    ("Shah Luthfullah Dargah", "Other", "Tembhurni", "Himayatnagar", "Nanded", "Final"),
    ("Sonapir Dargah (Mahur)", "Other", "Mahur", "Mahur", "Nanded", "Final"),
    ("Sarfaraj Khan Masjid", "Other", "Biloli", "Biloli", "Nanded", "Final"),
    ("Zainuddin Rafai Dargah", "Other", "Deglur", "Deglur", "Nanded", "Final"),
    ("Hazrat Sadaruddin and Badaruddin", "Other", "Shekapur", "Mahur", "Nanded", "Final"),
    ("Hindu Caves (Shiur)", "Caves", "Shiur", "Hadgaon", "Nanded", "Final"),
    ("Kshetrapal Temple Remains", "Archaeological Site", "Manaspuri", "Kandhar", "Nanded", "Final"),

    # ═══ NAGPUR DIVISION ═══
    # District Buldhana
    ("Kanchanicha Mahal", "Other", "Faijalapur", "Mehkar", "Buldhana", "First"),
    ("Nilkantheswar Temple (Sindhkhedraja)", "Temple", "Sindhkhedraja", "Sindhkhedraja", "Buldhana", "First"),
    ("Rangmahal (Sindhkhedraja)", "Other", "Sindhkhedraja", "Sindhkhedraja", "Buldhana", "Final"),
    ("Moti Talav", "Other", "Sindhkhedraja", "Sindhkhedraja", "Buldhana", "Final"),
    ("Lakhuji Jadhav Wada", "Other", "Sindhkhedraja", "Sindhkhedraja", "Buldhana", "Final"),
    ("Savkarwada", "Other", "Sindhkhedraja", "Sindhkhedraja", "Buldhana", "Final"),
    # District Akola
    ("Inayat Manzil", "Other", "Balapur", "Balapur", "Akola", "Final"),
    ("Darur Sharar Masjid", "Other", "Balapur", "Balapur", "Akola", "Final"),
    ("Darwaza Gate (Balapur)", "Other", "Balapur", "Balapur", "Akola", "Final"),
    ("Badi Roja Masjid", "Other", "Balapur", "Balapur", "Akola", "Final"),
    ("Jede Aala Dargah and Masjid", "Other", "Balapur", "Balapur", "Akola", "First"),
    # District Amravati
    ("Amba Gate (Amravati)", "Other", "Amravati", "Amravati", "Amravati", "First"),
    ("Jawahar Gate (Amravati)", "Other", "Amravati", "Amravati", "Amravati", "First"),
    ("Payryachi Vihir", "Other", "Mahimapur", "Daryapur", "Amravati", "Final"),
    ("Sasu Sunechi Vihir", "Other", "Riddhpur", "Morshi", "Amravati", "Final"),
    # District Washim
    ("Delhi Gate (Karanja Lad)", "Other", "Karanja Lad", "Karanja Lad", "Washim", "First"),
    ("Darwa Gate (Karanja Lad)", "Other", "Karanja Lad", "Karanja Lad", "Washim", "Final"),
    ("Poha Gate (Karanja Lad)", "Other", "Karanja Lad", "Karanja Lad", "Washim", "Final"),
    ("Mangrul Gate (Karanja Lad)", "Other", "Karanja Lad", "Karanja Lad", "Washim", "Final"),
    # District Nagpur
    ("Umred Fort", "Fort", "Umred", "Umred", "Nagpur", "Final"),
    ("Kapatram Temple", "Temple", "Gadmandir", "Ramtek", "Nagpur", "First"),
    ("Kapureshwar Temple", "Temple", "Katol", "Nagpur", "Nagpur", "First"),
    ("Keval Narsimha Temple", "Temple", "Gadmandir", "Ramtek", "Nagpur", "First"),
    ("Ganesh Temple (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Chandramauli Temple", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Chatri No. 3", "Other", "Ambala Talav", "Ramtek", "Nagpur", "First"),
    ("Chatri No. 4", "Other", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Chatri No. 19", "Other", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Chatri No. 24", "Other", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Chatri No. 26", "Other", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Jagannath Temple (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Datta Temple (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Datta Temple (Katol)", "Temple", "Katol", "Katol", "Nagpur", "Final"),
    ("Dharmashala No. 2951", "Other", "Ramtek", "Ramtek", "Nagpur", "Final"),
    ("Nagardhan Fort", "Fort", "Nagardhan", "Ramtek", "Nagpur", "Final"),
    ("Panchashikhari Temple", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Purushottam Maharaj Temple", "Temple", "Katol", "Nagpur", "Nagpur", "First"),
    ("Bhogram Temple", "Temple", "Gadmandir", "Ramtek", "Nagpur", "First"),
    ("Bholahudki Archaeological Site", "Archaeological Site", "Mandhal", "Umred", "Nagpur", "Final"),
    ("Mahadev Temple No. 6 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Mahadev Temple No. 32 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Ranisaheb Samadhi", "Other", "Ambala", "Ramtek", "Nagpur", "Final"),
    ("Renuka Temple (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "First"),
    ("Rudra Narsimha Temple", "Temple", "Gadmandir", "Ramtek", "Nagpur", "First"),
    ("Ram Ganesh Gadkari Monument", "Other", "Savner", "Savner", "Nagpur", "Final"),
    ("Ram Temple (Bajargaon)", "Temple", "Bajargaon", "Nagpur", "Nagpur", "First"),
    ("Ram Swami Temple", "Temple", "Gadmandir", "Ramtek", "Nagpur", "First"),
    ("Lakshman Swami Temple", "Temple", "Gadmandir", "Ramtek", "Nagpur", "First"),
    ("Varaha Temple (Gadmandir)", "Temple", "Gadmandir", "Ramtek", "Nagpur", "First"),
    ("Shiva Temple (Bajargaon)", "Temple", "Bajargaon", "Nagpur", "Nagpur", "First"),
    ("Shiva Temple No. 2 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 5 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "First"),
    ("Shiva Temple No. 7 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "First"),
    ("Shiva Temple No. 9 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 10 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 11 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 12 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 17 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 27 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 30 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Shiva Temple No. 31 (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Saraswati Temple and Kund (Katol)", "Temple", "Katol", "Katol", "Nagpur", "Final"),
    ("Vitthal Temple (Mulmule)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Harihar Temple (Ambala Talav)", "Temple", "Ambala Talav", "Ramtek", "Nagpur", "Final"),
    ("Harihar Samadhi Temple (Ambora)", "Other", "Ambora", "Kuhi", "Nagpur", "First"),
    # District Yavatmal
    ("Kedareshwar Temple (Yavatmal)", "Temple", "Azaad Maidan", "Yavatmal", "Yavatmal", "First"),
    ("Mahadev Temple (Pusad)", "Temple", "Pusad", "Yavatmal", "Yavatmal", "First"),
    # District Bhandara
    ("Ambagad Fort", "Fort", "Ambagad", "Tumsar", "Bhandara", "First"),
    ("Gosavi Samadhi", "Other", "Mendha Ward", "Bhandara", "Bhandara", "Final"),
    ("Murlidhar Temple (Pavni)", "Temple", "Pavni", "Pavni", "Bhandara", "Final"),
    # District Chandrapur
    ("Garud Stambha (Manikgad)", "Other", "Manikgad", "Jivati", "Chandrapur", "Final"),
    ("Rushi Talav Cave", "Caves", "Bhatala", "Varora", "Chandrapur", "Final"),
    ("Prehistoric Site (Manikgad)", "Archaeological Site", "Manikgad", "Jivati", "Chandrapur", "Final"),
    ("Bhavani Temple (Bhatala)", "Temple", "Bhatala", "Varora", "Chandrapur", "Final"),
    ("Manikgad Fort", "Fort", "Manikgad", "Jivati", "Chandrapur", "Final"),
    ("Mahadev Temple (Babupeth)", "Temple", "Babupeth", "Chandrapur", "Chandrapur", "Final"),
    ("Mahadev Temple (Bhatala)", "Temple", "Bhatala", "Varora", "Chandrapur", "First"),
    ("Vishnu Temple (Manikgad)", "Temple", "Manikgad", "Jivati", "Chandrapur", "Final"),
    ("Shankar Temple (Bhisi)", "Temple", "Bhisi", "Chimur", "Chandrapur", "Final"),
    ("Someshwar Temple (Rajura)", "Temple", "Rajura", "Rajura", "Chandrapur", "Final"),
    # District Gondia
    ("Kalbhairav Temple (Nagara)", "Temple", "Nagara", "Gondia", "Gondia", "First"),
]


def normalize_district(district: str) -> str:
    """Normalize district name for filename."""
    return district.lower().replace(" ", "_")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Group by district
    by_district: dict[str, list] = {}
    for row in MONUMENTS:
        name, mtype, place, taluka, district, notif_status = row
        by_district.setdefault(district, []).append(row)

    total = 0
    for district in sorted(by_district.keys()):
        rows = by_district[district]
        filename = f"{normalize_district(district)}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "monument_type", "place", "taluka", "district",
                "notification_status", "lat", "lon", "geocode_status",
            ])
            for name, mtype, place, taluka, dist, notif in rows:
                writer.writerow([
                    name, mtype, place, taluka, dist, notif,
                    "", "", "pending",
                ])

        total += len(rows)
        print(f"  {filename}: {len(rows)} monuments")

    print(f"\nTotal: {total} monuments across {len(by_district)} districts")


if __name__ == "__main__":
    main()
