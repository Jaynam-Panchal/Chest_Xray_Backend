from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
import io
import time
import os
import re
from groq import Client
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import shutil
load_dotenv()

PORT = int(os.environ.get('PORT', 5000))
HUGGINGFACE_REPO = os.getenv('HUGGINGFACE_REPO', 'jainam1510/Chest_Xray_model')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

if HUGGINGFACE_TOKEN is None:
    print('HUGGINGFACE_TOKEN not set - public repo access only or set token for private repo')


def download_model_from_hf(filename):
    """Download a file from the Hugging Face repo into local `best_models/` if missing.
    Returns the local path to the downloaded file (inside `best_models/`) or the cached path.
    """
    repo_id = HUGGINGFACE_REPO
    token = HUGGINGFACE_TOKEN
    candidates = [filename, os.path.join('best_models', filename)]
    last_err = None
    for candidate in candidates:
        try:
            print(f"Attempting to download '{candidate}' from HF repo '{repo_id}'")
            cached_file = hf_hub_download(repo_id=repo_id, filename=candidate, token=token)
            print(f"Downloaded model to cache: {cached_file}")

            # Ensure local best_models folder exists
            os.makedirs('best_models', exist_ok=True)
            target_path = os.path.join('best_models', os.path.basename(filename))

            try:
                shutil.copy(cached_file, target_path)
                print(f"Copied model to local path: {target_path}")
                return target_path
            except Exception as e:
                print(f"Failed to copy cached model to {target_path}: {e}")
                return cached_file
        except Exception as e:
            print(f"Could not download '{candidate}': {e}")
            last_err = e
    raise FileNotFoundError(f"Could not find '{filename}' in HF repo '{repo_id}': {last_err}")


def predownload_all_models():
    """Attempt to download all models listed in MODEL_PATHS into `best_models/`.
    This runs at startup on Render to ensure files are in the container filesystem.
    """
    os.makedirs('best_models', exist_ok=True)
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            try:
                print(f"Prefetching model '{name}' -> '{path}'")
                downloaded = download_model_from_hf(path)
                print(f"Model '{name}' available at: {downloaded}")
            except Exception as e:
                print(f"Warning: failed to prefetch model '{name}': {e}")

app = Flask(__name__)

# CORS configuration - Update with your frontend URL after deployment
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://chest-xray.vercel.app",  # Update this
    os.environ.get('FRONTEND_URL', '')  # Add from env variable
]

# Filter out empty strings
ALLOWED_ORIGINS = [origin for origin in ALLOWED_ORIGINS if origin]

CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# Request counter for variation
request_counter = 0

# ============ CONFIGURATION ============
MODEL_PATHS = {
    "VGG19": "OVERALL_BEST_VGG19_lr0.005_weight_decay0.0001_batch_size32_optimizersgd.pth",
    "ResNet18": "BEST_ResNet18_lr0.01_weight_decay0.0001_batch_size32_optimizersgd.pth",
    "ViT_Small": "BEST_ViT_Small_lr0.0001_weight_decay0.05_batch_size32_optimizeradamw.pth"
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables!")

# =============================================================

NUM_CLASSES = 4
CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE = 5.0

print(f"Using device: {DEVICE}")
print(f"Python version: {os.sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Initialize Groq client
if GROQ_API_KEY:
    groq_client = Client(api_key=GROQ_API_KEY)
else:
    groq_client = None

# Store loaded models in memory
loaded_models = {}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =============================================================
#                   MODEL BUILDERS
# =============================================================

def get_resnet18(num_classes):
    """Build ResNet18 model"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vgg19(num_classes):
    """Build VGG19 model"""
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def get_vit_small(num_classes):
    """Build Vision Transformer Small model"""
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# =============================================================
#         MEDICAL REPORT GENERATION USING GROQ
# =============================================================

def generate_medical_report(predicted_class, confidence, all_probabilities, request_id=0):
    """Generate medical report using Groq API"""
    if not groq_client:
        print("Groq client not initialized, using fallback reports")
        confidence_tier = get_confidence_tier(confidence)
        return {
            'clinical_interpretation': get_fallback_clinical(predicted_class, confidence),
            'recommendations': get_fallback_recommendations(predicted_class, confidence_tier)
        }
    
    try:
        import random
        import hashlib
        
        probs_text = ", ".join([f"{name}: {prob*100:.1f}%" for name, prob in sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)])
        
        # Confidence Tier System
        confidence_tier = get_confidence_tier(confidence)
        
        if confidence_tier == "very_high":
            confidence_descriptor = "high confidence, clear and definitive findings"
            severity_guidance = "findings are very clear and unmistakable"
            num_findings = 4
            num_recommendations = 6
        elif confidence_tier == "high":
            confidence_descriptor = "good confidence, typical findings observed"
            severity_guidance = "findings are evident and consistent with diagnosis"
            num_findings = 4
            num_recommendations = 5
        elif confidence_tier == "moderate":
            confidence_descriptor = "moderate confidence, suggestive findings"
            severity_guidance = "findings suggest diagnosis but further correlation advised"
            num_findings = 4
            num_recommendations = 5
        else:
            confidence_descriptor = "lower confidence, subtle findings"
            severity_guidance = "findings are subtle, clinical correlation strongly recommended"
            num_findings = 3
            num_recommendations = 4
        
        # Generate variation seed
        confidence_bucket = f"{confidence:.4f}"
        variation_string = f"{predicted_class}_{confidence_bucket}_{request_id}_{int(time.time() * 1000) % 10000}"
        variation_seed = int(hashlib.md5(variation_string.encode()).hexdigest()[:8], 16) % 10000
        random.seed(variation_seed)
        
        # Get class-specific context
        context = get_class_context(predicted_class)
        
        # Select tier-specific actions
        tier_actions = get_tier_actions(context, confidence_tier)
        
        # Determine explanation style
        explanation_style = get_explanation_style(confidence_tier)
        
        # Terminology variants
        terminology_variants = {
            'ggo_terms': ['Bilat. ground-glass opacities', 'GGOs present bilaterally', 'Hazy infiltrates both lungs', 
                         'Ground-glass changes evident', 'Patchy opacities observed', 'Diffuse GGO pattern'],
            'distribution': ['Peripheral distribution noted', 'Periph. pattern evident', 'Outer lung zones affected',
                           'Lateral predominance seen', 'Subpleural distribution', 'Peripheral-predominant changes'],
            'verbs': ['seen', 'noted', 'observed', 'present', 'identified', 'evident', 'detected', 'visible'],
            'both_lungs': ['both lungs', 'bilaterally', 'bilateral lung fields', 'affecting both sides', 'in both lung zones']
        }
        
        selected_variants = {
            'ggo': terminology_variants['ggo_terms'][variation_seed % len(terminology_variants['ggo_terms'])],
            'dist': terminology_variants['distribution'][variation_seed % len(terminology_variants['distribution'])],
            'verb': terminology_variants['verbs'][variation_seed % len(terminology_variants['verbs'])],
            'lungs': terminology_variants['both_lungs'][variation_seed % len(terminology_variants['both_lungs'])]
        }
        
        # Build prompt
        prompt = build_report_prompt(
            predicted_class, confidence, confidence_descriptor, confidence_tier,
            variation_seed, context, tier_actions, num_findings, num_recommendations,
            explanation_style, selected_variants
        )
        
        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a radiologist. Use medical terminology and patient-friendly language.\nReport ID: {variation_seed} | Tier: {confidence_tier}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.65,
            max_tokens=320,
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.4
        )
        
        report_text = chat_completion.choices[0].message.content.strip()
        print(f"\n=== RAW REPORT FROM GROQ ===\n{report_text}\n=== END RAW REPORT ===\n")
        
        # Clean and parse report
        clinical_text, recommendations_text = parse_report(
            report_text, predicted_class, confidence, confidence_tier,
            num_findings, num_recommendations
        )
            
        return {
            'clinical_interpretation': clinical_text,
            'recommendations': recommendations_text
        }

    except Exception as e:
        print(f"Groq API Error: {e}")
        import traceback
        traceback.print_exc()
        
        confidence_tier = get_confidence_tier(confidence)
        return {
            'clinical_interpretation': get_fallback_clinical(predicted_class, confidence),
            'recommendations': get_fallback_recommendations(predicted_class, confidence_tier)
        }

def get_confidence_tier(confidence):
    """Determine confidence tier based on score"""
    if confidence >= 0.90:
        return "very_high"
    elif confidence >= 0.75:
        return "high"
    elif confidence >= 0.60:
        return "moderate"
    else:
        return "low"

def get_class_context(predicted_class):
    """Get medical context for each class"""
    class_context = {
        'COVID': {
            'radiographic_findings': 'bilateral ground-glass opacities, peripheral distribution, typical COVID-19 pneumonia pattern',
            'clinical_significance': 'viral infection affecting both lungs, respiratory system involvement, contagious disease',
            'patient_explanation': 'cloudy patches in both lungs showing viral infection pattern typical of COVID-19',
            'actions_very_high': 'Immediate RT-PCR/COVID test required, strict home isolation mandatory, continuous oxygen monitoring, antiviral therapy initiation, public health notification',
            'actions_high': 'RT-PCR/COVID test recommended, home isolation advised, regular oxygen monitoring, consider antiviral therapy, standard follow-up',
            'actions_moderate': 'Consider RT-PCR testing, home precautions suggested, monitor symptoms, clinical correlation needed, conservative management',
            'actions_low': 'Clinical correlation essential, consider testing if symptoms worsen, observe closely, repeat imaging recommended, differential diagnosis needed',
            'monitoring': 'oxygen saturation, breathing difficulty, fever progression',
            'follow_up': 'chest X-ray in 5-7 days, telehealth check-ins, emergency if breathing worsens'
        },
        'Viral Pneumonia': {
            'radiographic_findings': 'bilateral interstitial infiltrates, diffuse pulmonary involvement, viral pneumonia pattern',
            'clinical_significance': 'non-COVID viral lung infection, inflammatory response in airways, respiratory compromise',
            'patient_explanation': 'widespread inflammation in both lungs caused by a viral infection (not COVID-19)',
            'actions_very_high': 'Comprehensive viral panel testing urgent, respiratory isolation mandatory, oxygen therapy if needed, broad antiviral coverage, intensive monitoring',
            'actions_high': 'Viral panel testing recommended, respiratory precautions advised, supportive care, symptom management, regular monitoring',
            'actions_moderate': 'Consider viral testing, supportive measures, rest and hydration, monitor progression, clinical correlation',
            'actions_low': 'Clinical assessment needed, observe symptoms, consider testing if worsening, supportive care, differential workup',
            'monitoring': 'temperature, breathing pattern, energy levels, cough progression',
            'follow_up': 'follow-up imaging in 10 days, symptom diary, gradual activity increase'
        },
        'Lung_Opacity': {
            'radiographic_findings': 'pulmonary opacity present, localized or diffuse density, etiology unclear',
            'clinical_significance': 'abnormal area in lung tissue, requires further investigation, could indicate infection or inflammation',
            'patient_explanation': 'unclear cloudy area in the lungs that needs more tests to understand the cause',
            'actions_very_high': 'Urgent CT scan required, comprehensive blood work, immediate pulmonology consult, rule out serious pathology, expedited workup',
            'actions_high': 'CT scan recommended for characterization, blood work advised, pulmonology referral, thorough investigation needed',
            'actions_moderate': 'Consider CT for detail, basic labs, clinical correlation, monitor progression, conservative approach',
            'actions_low': 'Observe closely, consider repeat imaging, clinical assessment, may be benign finding, watchful waiting',
            'monitoring': 'new symptoms, breathing changes, chest discomfort, cough development',
            'follow_up': 'repeat chest X-ray in 2-4 weeks, track symptoms daily, specialist referral if needed'
        },
        'Normal': {
            'radiographic_findings': 'clear bilateral lung fields, normal cardiopulmonary silhouette, no acute abnormalities',
            'clinical_significance': 'no evidence of infection, inflammation, or structural abnormality, healthy lung appearance',
            'patient_explanation': 'lungs appear healthy and clear with no signs of infection or disease',
            'actions_very_high': 'No intervention required, routine health maintenance, continue normal activities, annual check-ups',
            'actions_high': 'No immediate treatment needed, routine preventive care, standard follow-up schedule',
            'actions_moderate': 'Likely no intervention needed, observe if symptoms present, routine care advised',
            'actions_low': 'Clinical correlation recommended, monitor symptoms, repeat imaging if concerns arise',
            'monitoring': 'routine health check-ups, watch for new symptoms if they develop',
            'follow_up': 'annual wellness visits, chest X-ray only if symptoms appear, maintain healthy lifestyle'
        }
    }
    return class_context.get(predicted_class, class_context['Normal'])

def get_tier_actions(context, confidence_tier):
    """Get tier-specific actions"""
    actions_key = f'actions_{confidence_tier}'
    return context.get(actions_key, context.get('actions_high', ''))

def get_explanation_style(confidence_tier):
    """Determine explanation style based on confidence tier"""
    if confidence_tier in ["very_high", "high"]:
        return "minimal"
    elif confidence_tier == "moderate":
        return "mixed"
    else:
        return "moderate"

def build_report_prompt(predicted_class, confidence, confidence_descriptor, 
                       confidence_tier, variation_seed, context, tier_actions,
                       num_findings, num_recommendations, explanation_style, 
                       selected_variants):
    """Build the prompt for Groq API"""
    return f"""You are a radiologist writing quick clinical notes. Write naturally like a real doctor - not too polished, mix of formal/casual language.

DIAGNOSIS: {predicted_class}
CONFIDENCE: {confidence*100:.1f}% ({confidence_descriptor})
CONFIDENCE TIER: {confidence_tier}
REPORT NUMBER: {variation_seed}

Medical Context: {context['radiographic_findings']}
Clinical Significance: {context['clinical_significance']}
Actions Needed for {confidence_tier.upper()} confidence: {tier_actions}
Monitoring: {context['monitoring']}
Follow-up: {context['follow_up']}

CRITICAL: This is report #{variation_seed}. You MUST write COMPLETELY DIFFERENT from any previous reports.
- Use DIFFERENT starting phrases for each bullet
- VARY terminology choices significantly
- DO NOT repeat patterns from earlier reports
- Think of 5 different ways to say the same medical concept
- Try using these varied terms: "{selected_variants['ggo']}", "{selected_variants['dist']}", "{selected_variants['verb']}", "{selected_variants['lungs']}"

Write TWO sections in a natural doctor's style:

SECTION 1 - FINDINGS (EXACTLY {num_findings} bullet points):
Write EXACTLY {num_findings} observations about what you see. Mix styles:
- Use medical terms directly sometimes (no explanation needed)
- Explain only when helpful for patient understanding
- Vary between technical and plain language
- Use abbreviations occasionally (bilat., diffuse, periph.)
- Each point: 4-8 words, natural phrasing

Style guidance: {explanation_style}

SECTION 2 - RECOMMENDATIONS (EXACTLY {num_recommendations} bullet points):
Write EXACTLY {num_recommendations} action items. Be practical and direct:
- Mix formal and casual tone
- Use medical abbreviations sometimes (F/U, O2 sats, CXR, CT)
- Vary detail level (some detailed, some brief)
- Real doctor language: "Recommend", "Consider", "Advise", "Should get"
- Each point: 4-9 words

Format with bullets (•), no headers, blank line between sections."""

def parse_report(report_text, predicted_class, confidence, confidence_tier,
                num_findings, num_recommendations):
    """Parse and clean the Groq report response"""
    # Clean unwanted text
    report_text = re.sub(r'\*\*', '', report_text)
    report_text = re.sub(r'(?i)(WHAT WE FOUND.*?:?\s*)', '', report_text)
    report_text = re.sub(r'(?i)(WHAT YOU SHOULD DO.*?:?\s*)', '', report_text)
    report_text = re.sub(r'(?i)(CLINICAL.*?:?\s*)', '', report_text)
    report_text = re.sub(r'(?i)(RECOMMENDATIONS?.*?:?\s*)', '', report_text)
    report_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', report_text)
    report_text = re.sub(r'(\w)(needed|required|recommended)', r'\1 \2', report_text)
    
    sections_split = re.split(r'\n\s*\n+', report_text.strip())
    
    clinical_text = ''
    recommendations_text = ''
    
    if len(sections_split) >= 2:
        clinical_text = sections_split[0].strip()
        recommendations_text = sections_split[1].strip()
    else:
        all_lines = [l.strip() for l in report_text.split('\n') if l.strip()]
        bullets = [l for l in all_lines if l.startswith('•') or l.startswith('-')]
        
        if len(bullets) >= (num_findings + num_recommendations):
            clinical_text = '\n'.join(bullets[:num_findings])
            recommendations_text = '\n'.join(bullets[num_findings:num_findings+num_recommendations])
        elif len(bullets) >= num_findings:
            clinical_text = '\n'.join(bullets[:num_findings])
            recommendations_text = '\n'.join(bullets[num_findings:])
        else:
            return (
                get_fallback_clinical(predicted_class, confidence),
                get_fallback_recommendations(predicted_class, confidence_tier)
            )
    
    if not clinical_text or len(clinical_text) < 20:
        clinical_text = get_fallback_clinical(predicted_class, confidence)
    if not recommendations_text or len(recommendations_text) < 20:
        recommendations_text = get_fallback_recommendations(predicted_class, confidence_tier)
        
    return clinical_text, recommendations_text

def get_fallback_clinical(predicted_class, confidence):
    """Fallback clinical interpretations"""
    interpretations = {
        'COVID': '''• Bilateral ground-glass opacities (cloudy patches both lungs)
• Peripheral distribution pattern consistent with COVID-19
• Moderate to severe respiratory system involvement
• Viral pneumonia characteristics highly suggestive of COVID''',
        
        'Viral Pneumonia': '''• Bilateral interstitial infiltrates (inflammation in both lungs)
• Diffuse pulmonary involvement indicating viral infection
• Non-COVID viral pneumonia pattern identified
• Respiratory compromise with inflammatory changes present''',
        
        'Lung_Opacity': '''• Pulmonary opacity detected requiring further workup
• Localized density increase of uncertain etiology
• Could indicate early infection or inflammatory process
• Additional imaging needed for definitive characterization''',
        
        'Normal': '''• Clear bilateral lung fields with normal appearance
• No radiographic evidence of acute abnormality
• Cardiopulmonary silhouette within normal limits
• Reassuring findings with no pathological changes'''
    }
    return interpretations.get(predicted_class, interpretations['Normal'])

def get_fallback_recommendations(predicted_class, confidence_tier="high"):
    """Fallback recommendations based on diagnosis and confidence"""
    recommendations = {
        'COVID': {
            'very_high': '''• Immediate RT-PCR test required for COVID-19 confirmation
• Strict home isolation mandatory to prevent transmission
• Continuous oxygen monitoring with pulse oximeter essential
• Antiviral therapy initiation should be considered urgently
• Emergency follow-up chest imaging in 5 days
• Public health notification and contact tracing required''',
            'high': '''• RT-PCR test for COVID-19 confirmation recommended
• Home isolation advised to prevent transmission
• Regular oxygen saturation monitoring using pulse oximeter
• Consider antiviral therapy based on clinical presentation
• Follow-up chest imaging in 7 days
• Standard contact tracing procedures''',
            'moderate': '''• Consider RT-PCR testing if symptoms progress
• Home precautions and social distancing suggested
• Monitor oxygen levels if respiratory symptoms worsen
• Clinical correlation needed before treatment decisions
• Follow-up imaging in 10-14 days recommended''',
            'low': '''• Clinical assessment essential before testing
• Observe closely for symptom development
• Consider repeat imaging for confirmation
• Differential diagnosis workup recommended'''
        },
        'Viral Pneumonia': {
            'very_high': '''• Comprehensive viral panel testing urgent for pathogen identification
• Respiratory isolation precautions mandatory
• Oxygen therapy required if saturation below 94%
• Broad antiviral coverage consideration
• Intensive monitoring for bacterial superinfection
• Follow-up chest X-ray in 7 days essential''',
            'high': '''• Viral panel testing recommended to identify pathogen
• Respiratory isolation precautions advised
• Oxygen therapy if saturation drops below 94%
• Supportive care with rest and hydration
• Monitor for bacterial superinfection development''',
            'moderate': '''• Consider viral testing if symptoms worsen
• Supportive measures and symptom management
• Monitor oxygen levels and breathing pattern
• Rest and adequate hydration important''',
            'low': '''• Clinical assessment needed for diagnosis confirmation
• Observe symptom progression carefully
• Consider testing if clinical picture worsens
• Supportive care measures recommended'''
        },
        'Lung_Opacity': {
            'very_high': '''• Urgent CT chest required for detailed characterization
• Comprehensive blood work including CBC, CRP mandatory
• Immediate pulmonology consultation needed
• Rule out serious pathology promptly
• Expedited diagnostic workup essential
• Short-interval follow-up imaging in 1-2 weeks''',
            'high': '''• CT chest recommended for detailed characterization
• Laboratory workup including CBC and inflammatory markers
• Pulmonology consultation for further evaluation
• Thorough investigation of underlying cause needed
• Follow-up imaging in 2-4 weeks''',
            'moderate': '''• Consider CT scan for additional detail
• Basic laboratory tests may be helpful
• Clinical correlation strongly recommended
• Monitor for symptom development''',
            'low': '''• Observe closely without immediate intervention
• Clinical assessment to correlate findings
• May represent benign or artifact finding
• Consider repeat imaging only if symptoms develop'''
        },
        'Normal': {
            'very_high': '''• No medical intervention required
• Continue routine preventive health maintenance
• Annual wellness check-ups as scheduled
• Maintain healthy lifestyle practices''',
            'high': '''• No immediate treatment needed
• Routine preventive care recommended
• Standard follow-up schedule appropriate
• Continue normal activities''',
            'moderate': '''• Likely no intervention needed
• Observe if any symptoms present
• Routine care advised''',
            'low': '''• Clinical correlation recommended for certainty
• Monitor for any symptom development
• Consider repeat imaging if concerns arise'''
        }
    }
    
    diagnosis_recs = recommendations.get(predicted_class, recommendations['Normal'])
    return diagnosis_recs.get(confidence_tier, diagnosis_recs.get('high', ''))

# =============================================================
#                   MODEL LOADING FUNCTIONS
# =============================================================

def load_model(model_type):
    """Load a specific model by type"""
    if model_type in loaded_models:
        print(f"Using cached {model_type} model")
        return loaded_models[model_type]
    
    print(f"Loading {model_type} model...")
    
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_path = MODEL_PATHS[model_type]
    
    # If the model file is not present locally, try to download from Hugging Face repo
    if not os.path.exists(model_path):
        try:
            def download_model_from_hf(filename):
                repo_id = HUGGINGFACE_REPO
                token = HUGGINGFACE_TOKEN
                # Try the exact filename in the repo, and also under best_models/
                candidates = [filename, os.path.join('best_models', filename)]
                last_err = None
                for candidate in candidates:
                    try:
                        print(f"Attempting to download '{candidate}' from HF repo '{repo_id}'")
                        cached_file = hf_hub_download(repo_id=repo_id, filename=candidate, token=token)
                        print(f"Downloaded model to cache: {cached_file}")

                        # Ensure local best_models folder exists
                        target_path = filename if os.path.dirname(filename) else os.path.join('best_models', filename)
                        os.makedirs(os.path.dirname(target_path) or '.', exist_ok=True)

                        # Copy cached file to repository-local path so the server has a stable copy
                        try:
                            shutil.copy(cached_file, target_path)
                            print(f"Copied model to local path: {target_path}")
                            return target_path
                        except Exception as e:
                            print(f"Failed to copy cached model to {target_path}: {e}")
                            # Fallback to returning the cached path
                            return cached_file
                    except Exception as e:
                        print(f"Could not download '{candidate}': {e}")
                        last_err = e
                raise FileNotFoundError(f"Could not find '{filename}' in HF repo '{repo_id}': {last_err}")

            print(f"Model file '{model_path}' not found locally. Trying Hugging Face repo...")
            model_path = download_model_from_hf(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Model file not found locally and HF download failed: {e}")
    
    # Build the appropriate model architecture
    if model_type == "ResNet18":
        model = get_resnet18(NUM_CLASSES)
    elif model_type == "VGG19":
        model = get_vgg19(NUM_CLASSES)
    elif model_type == "ViT_Small":
        model = get_vit_small(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    # Note: remove unsupported kwargs when using torch.load
    state = torch.load(model_path, map_location=DEVICE)
    # If the saved object is a state_dict, load it. Otherwise, try to load model directly.
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        # If a full model was saved, attempt to load state_dict attribute or assign directly
        try:
            model.load_state_dict(state.state_dict())
        except Exception:
            # As a last resort, replace model object
            model = state
    model = model.to(DEVICE)
    model.eval()
    
    # Cache the model
    loaded_models[model_type] = model
    print(f"{model_type} model loaded successfully!")
    
    return model

# Prefetch models (Render-friendly) and preload VGG19 at startup (most commonly used)
try:
    predownload_all_models()
except Exception as e:
    print(f"Warning: predownload_all_models failed: {e}")

try:
    print("Preloading VGG19 model...")
    load_model("VGG19")
except Exception as e:
    print(f"WARNING: Failed to preload VGG19: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    global request_counter
    request_counter += 1
    current_request_id = request_counter
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get model type from request (default to VGG19)
        model_type = request.form.get('model', 'VGG19')
        
        if model_type not in MODEL_PATHS:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400
        
        # Load the requested model
        try:
            model = load_model(model_type)
        except Exception as e:
            return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Prediction
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs / TEMPERATURE, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        processing_time = int((time.time() - start_time) * 1000)

        predicted_label = CLASS_NAMES[predicted_class.item()]
        confidence_score = confidence.item()
        all_probabilities = {CLASS_NAMES[i]: float(probabilities[0][i].item()) for i in range(NUM_CLASSES)}

        # Generate medical report
        medical_report = generate_medical_report(
            predicted_label, 
            confidence_score, 
            all_probabilities, 
            current_request_id
        )

        result = {
            'predicted_class': predicted_label,
            'confidence': confidence_score,
            'probabilities': all_probabilities,
            'processing_time': processing_time,
            'model_name': model_type,
            'medical_report': medical_report
        }

        print(f"Prediction ({model_type}): {predicted_label} ({confidence_score:.2%})")
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =============================================================
#                      SERVER START
# =============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("COVID-19 X-RAY CLASSIFICATION SERVER")
    print("=" * 70)
    print(f"Available Models: {list(MODEL_PATHS.keys())}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Python: {os.sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)
    print(f"\nServer running on http://0.0.0.0:{PORT}")
    print("Press CTRL+C to stop\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)