import numpy as np
import re
import string
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# Configure logging
logger = logging.getLogger(__name__)

# Initialize NLTK stop words and stemmer
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    STOP_WORDS = set(stopwords.words('english'))
    logger.info(f"Loaded {len(STOP_WORDS)} NLTK stop words")
except LookupError:
    # Download required NLTK data
    logger.info("Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    STOP_WORDS = set(stopwords.words('english'))
    logger.info(f"Downloaded and loaded {len(STOP_WORDS)} NLTK stop words")

# Initialize stemmer
stemmer = PorterStemmer()

# Cooking-specific stop words (measurements, units, and common non-meaningful words)
COOKING_STOP_WORDS = {
    'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'teaspoon', 'teaspoons', 'tsp',
    'ounce', 'ounces', 'oz', 'pound', 'pounds', 'lb', 'lbs', 'gram', 'grams', 'g',
    'kilogram', 'kilograms', 'kg', 'milliliter', 'milliliters', 'ml', 'liter', 'liters', 'l',
    'pinch', 'dash', 'clove', 'cloves', 'piece', 'pieces', 'slice', 'slices',
    'large', 'medium', 'small', 'whole', 'half', 'quarter', 'halved', 'quartered',
    'to', 'taste', 'needed', 'optional', 'divided', 'plus', 'more'
}

# Combine stop words
ALL_STOP_WORDS = STOP_WORDS | COOKING_STOP_WORDS

# Keep meaningful culinary adjectives that should NOT be filtered
KEEP_WORDS = {
    'fresh', 'dried', 'frozen', 'canned', 'raw', 'cooked', 'roasted', 'grilled',
    'ground', 'chopped', 'diced', 'minced', 'sliced', 'crushed', 'grated',
    'organic', 'kosher', 'sea', 'extra', 'virgin', 'unsalted', 'salted'
}

# Final stop words set (exclude words we want to keep)
FINAL_STOP_WORDS = ALL_STOP_WORDS - KEEP_WORDS

# Load cooking phrases from JSON file
import json
import os

def load_cooking_phrases():
    """Load cooking phrases from JSON file."""
    phrases_file = os.path.join(os.path.dirname(__file__), 'cooking_phrases.json')
    with open(phrases_file, 'r') as f:
        phrases = json.load(f)
        logger.info(f"Loaded {len(phrases)} cooking phrases from {phrases_file}")
        return set(phrases)

# Load cooking phrases at module initialization
COOKING_PHRASES = load_cooking_phrases()


def stem_token(token):
    """Apply Porter stemming to a single token."""
    try:
        return stemmer.stem(token)
    except Exception:
        return token


def extract_phrases(text):
    """
    Extract known cooking phrases from text.
    Returns a list of phrases found in the text.
    """
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    found_phrases = []
    
    for phrase in COOKING_PHRASES:
        if phrase in text_lower:
            found_phrases.append(phrase.replace(' ', '_'))  # Replace space with underscore
    
    return found_phrases


def normalize_text(text, apply_stemming=True, extract_phrases_flag=True):
    """
    Normalize text for ingredient matching with stemming and phrase extraction:
    - Extract multi-word cooking phrases (e.g., "olive oil" -> "olive_oil")
    - Convert to lowercase
    - Remove punctuation
    - Split into tokens
    - Apply stemming (optional)
    - Remove stop words (English + cooking-specific)
    - Keep meaningful culinary adjectives
    - Filter out very short tokens (< 2 chars)
    - Filter out pure numbers
    
    Args:
        text: Input text string
        apply_stemming: Whether to apply Porter stemming
        extract_phrases_flag: Whether to extract multi-word phrases
    
    Returns:
        Set of normalized tokens
    """
    if not isinstance(text, str):
        return set()
    
    tokens = set()
    
    # Extract phrases first (before removing punctuation)
    if extract_phrases_flag:
        phrases = extract_phrases(text)
        tokens.update(phrases)
    
    # Remove punctuation and convert to lowercase
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Split into tokens, remove stop words, short tokens, and numbers
    word_tokens = text.split()
    
    for token in word_tokens:
        # Skip if stop word, too short, or pure number
        if token in FINAL_STOP_WORDS or len(token) < 2 or token.isdigit():
            continue
        
        # Apply stemming if requested
        if apply_stemming:
            token = stem_token(token)
        
        tokens.add(token)
    
    return tokens


def normalize_text_for_tfidf(text):
    """
    Normalize text specifically for TF-IDF vectorization.
    Returns a space-separated string of normalized tokens.
    """
    tokens = normalize_text(text, apply_stemming=True, extract_phrases_flag=True)
    return ' '.join(sorted(tokens))  # Sort for consistency


def prepare_text_features(dataframe):
    """
    Prepare text features from RecipeIngredientParts using TF-IDF.
    Applies stemming and phrase extraction.
    
    Returns:
        TF-IDF matrix and fitted vectorizer
    """
    # Normalize ingredient text
    ingredient_texts = dataframe['RecipeIngredientParts'].apply(normalize_text_for_tfidf)
    
    # Create TF-IDF vectorizer with parameters optimized for ingredients
    tfidf_vectorizer = TfidfVectorizer(
        max_features=500,  # Limit vocabulary size
        min_df=2,          # Ignore terms that appear in less than 2 documents
        max_df=0.8,        # Ignore terms that appear in more than 80% of documents
        ngram_range=(1, 2),  # Use unigrams and bigrams for phrase detection
        lowercase=True,
        token_pattern=r'\b\w+\b'  # Match word tokens including underscore-connected phrases
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(ingredient_texts)
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}, features: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    return tfidf_matrix, tfidf_vectorizer


def scaling(dataframe, use_tfidf=True):
    """
    Scale nutritional features and optionally combine with TF-IDF text features.
    
    Args:
        dataframe: Input dataframe
        use_tfidf: Whether to include TF-IDF features from ingredients
    
    Returns:
        Combined feature matrix, scaler, and tfidf_vectorizer (or None)
    """
    # Scale nutritional features (columns 6-15)
    scaler = StandardScaler()
    nutrition_data = scaler.fit_transform(dataframe.iloc[:, 6:15].to_numpy())
    
    if use_tfidf:
        # Add TF-IDF features from ingredients
        tfidf_matrix, tfidf_vectorizer = prepare_text_features(dataframe)
        
        # Combine nutrition features with TF-IDF features
        # Convert nutrition to sparse matrix for efficient concatenation
        nutrition_sparse = csr_matrix(nutrition_data)
        combined_features = hstack([nutrition_sparse, tfidf_matrix])
        
        logger.info(f"Combined features shape: {combined_features.shape} (nutrition: {nutrition_data.shape[1]}, text: {tfidf_matrix.shape[1]})")
        return combined_features, scaler, tfidf_vectorizer
    else:
        return nutrition_data, scaler, None


def nn_predictor(prep_data):
    """
    Create and fit a NearestNeighbors model.
    Now supports both dense and sparse matrices.
    """
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    return neigh


def build_pipeline(neigh, scaler, tfidf_vectorizer, params):
    """
    Build prediction pipeline.
    
    Args:
        neigh: Fitted NearestNeighbors model
        scaler: Fitted StandardScaler for nutrition features
        tfidf_vectorizer: Fitted TfidfVectorizer (or None if not using text features)
        params: Parameters for kneighbors (n_neighbors, return_distance)
    """
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    
    # Store preprocessing objects for later use
    pipeline = Pipeline([('NN', transformer)])
    pipeline.scaler = scaler
    pipeline.tfidf_vectorizer = tfidf_vectorizer
    
    return pipeline

def extract_data(dataframe,ingredients):
    extracted_data=dataframe.copy()
    extracted_data=extract_ingredient_filtered_data(extracted_data,ingredients)
    return extracted_data
    
def extract_ingredient_filtered_data(dataframe, ingredients):
    """
    Filter dataframe by ingredients using token-based matching.
    
    Args:
        dataframe: DataFrame with 'RecipeIngredientParts' column
        ingredients: List of ingredient strings to match
    
    Returns:
        Filtered DataFrame containing recipes with ALL requested ingredients
    """
    if not ingredients:
        return dataframe.copy()
    
    extracted_data = dataframe.copy()
    
    # Normalize the requested ingredients into tokens
    requested_tokens = set()
    for ingredient in ingredients:
        requested_tokens.update(normalize_text(ingredient))
    
    if not requested_tokens:
        return extracted_data
    
    # Filter function: check if all requested tokens are present in recipe ingredients
    def has_all_ingredients(recipe_ingredients):
        recipe_tokens = normalize_text(recipe_ingredients)
        # Check if all requested tokens are in the recipe tokens
        return requested_tokens.issubset(recipe_tokens)
    
    # Apply the filter
    mask = extracted_data['RecipeIngredientParts'].apply(has_all_ingredients)
    extracted_data = extracted_data[mask]
    
    return extracted_data

def apply_pipeline(pipeline, _input, extracted_data, use_tfidf=True):
    """
    Apply the prediction pipeline to input nutrition values.
    
    Args:
        pipeline: Fitted pipeline
        _input: Nutrition input array (9 values)
        extracted_data: Filtered dataframe
        use_tfidf: Whether TF-IDF features were used
    """
    _input = np.array(_input).reshape(1, -1)
    
    if use_tfidf and hasattr(pipeline, 'tfidf_vectorizer') and pipeline.tfidf_vectorizer:
        # Need to create a dummy text feature vector (mean of all recipes)
        # This is a placeholder - ideally user would provide ingredient preferences too
        tfidf_vectorizer = pipeline.tfidf_vectorizer
        
        # Create zero vector for text features (no user ingredient text input)
        # Alternative: use mean of all recipes' TF-IDF vectors
        n_text_features = len(tfidf_vectorizer.get_feature_names_out())
        text_features = csr_matrix(np.zeros((1, n_text_features)))
        
        # Scale nutrition input
        scaled_nutrition = pipeline.scaler.transform(_input)
        scaled_nutrition_sparse = csr_matrix(scaled_nutrition)
        
        # Combine features
        combined_input = hstack([scaled_nutrition_sparse, text_features])
        
        indices = pipeline.transform(combined_input)[0]
    else:
        # Only nutrition features
        indices = pipeline.transform(_input)[0]
    
    return extracted_data.iloc[indices]


def pretrain_model(dataframe, use_tfidf=True):
    """
    Pre-train the TF-IDF vectorizer and NearestNeighbors model on the full dataset.
    This eliminates the need to retrain on every request.
    
    Args:
        dataframe: Full recipe dataframe
        use_tfidf: Whether to include TF-IDF features
    
    Returns:
        Dictionary containing pre-trained models and metadata
    """
    import time
    start = time.time()
    
    # Train on full dataset
    logger.info(f"Training on {len(dataframe)} recipes...")
    prep_data, scaler, tfidf_vectorizer = scaling(dataframe, use_tfidf=use_tfidf)
    
    # Fit nearest neighbors on all data
    neigh = nn_predictor(prep_data)
    
    elapsed = time.time() - start
    logger.info(f"Pre-training completed in {elapsed:.2f}s")
    
    # Return pre-trained components
    return {
        'features': prep_data,  # Pre-computed feature matrix
        'scaler': scaler,
        'tfidf_vectorizer': tfidf_vectorizer,
        'neigh': neigh,
        'use_tfidf': use_tfidf,
        'feature_dim': prep_data.shape[1],
        'tfidf_vocab_size': len(tfidf_vectorizer.get_feature_names_out()) if tfidf_vectorizer else 0
    }


def recommend(dataframe, _input, ingredients=[], params={'n_neighbors': 5, 'return_distance': False}, use_tfidf=True, pretrained_models=None):
    """
    Recommend recipes based on nutritional input and optional ingredient filters.
    
    Args:
        dataframe: Recipe dataframe
        _input: Nutritional values (9 features)
        ingredients: List of required ingredients for filtering
        params: Parameters for nearest neighbors
        use_tfidf: Whether to use TF-IDF features from ingredients text
        pretrained_models: Pre-trained models from pretrain_model() (optional, for fast inference)
    
    Returns:
        DataFrame of recommended recipes
    """
    # If ingredients filter is provided, we need to retrain on the filtered subset
    if ingredients:
        logger.info(f"Filtering by ingredients: {ingredients}")
        extracted_data = extract_data(dataframe, ingredients)
        
        if extracted_data.shape[0] >= params['n_neighbors']:
            # Retrain on filtered data (necessary for ingredient filtering)
            prep_data, scaler, tfidf_vectorizer = scaling(extracted_data, use_tfidf=use_tfidf)
            neigh = nn_predictor(prep_data)
            pipeline = build_pipeline(neigh, scaler, tfidf_vectorizer, params)
            return apply_pipeline(pipeline, _input, extracted_data, use_tfidf=use_tfidf)
        else:
            logger.warning(f"Not enough recipes ({extracted_data.shape[0]}) matching ingredients filter")
            return None
    
    # Use pre-trained models for fast inference (no ingredient filter)
    elif pretrained_models is not None:
        logger.info("Using pre-trained models for fast inference")
        # Build pipeline with pre-trained components
        pipeline = build_pipeline(
            pretrained_models['neigh'],
            pretrained_models['scaler'],
            pretrained_models['tfidf_vectorizer'],
            params
        )
        # Apply to full dataset
        return apply_pipeline(pipeline, _input, dataframe, use_tfidf=pretrained_models['use_tfidf'])
    
    # Fallback: train on-demand (if no pre-trained models available)
    else:
        logger.warning("No pre-trained models available, training on-demand (slow)")
        extracted_data = dataframe.copy()
        
        if extracted_data.shape[0] >= params['n_neighbors']:
            prep_data, scaler, tfidf_vectorizer = scaling(extracted_data, use_tfidf=use_tfidf)
            neigh = nn_predictor(prep_data)
            pipeline = build_pipeline(neigh, scaler, tfidf_vectorizer, params)
            return apply_pipeline(pipeline, _input, extracted_data, use_tfidf=use_tfidf)
        else:
            return None

def extract_quoted_strings(s):
    # Find all the strings inside double quotes
    strings = re.findall(r'"([^"]*)"', s)
    # Join the strings with 'and'
    return strings

def output_recommended_recipes(dataframe):
    if dataframe is not None:
        output=dataframe.copy()
        output=output.to_dict("records")
        for recipe in output:
            recipe['RecipeIngredientParts']=extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions']=extract_quoted_strings(recipe['RecipeInstructions'])
    else:
        output=None
    return output

