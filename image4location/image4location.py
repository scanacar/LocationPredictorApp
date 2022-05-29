from venv import create
import pandas as pd
import torch
import clip
from PIL import Image
import pandas
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import urllib.request
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")
st.showWarningOnDirectExecution = False

st.markdown("<h1 style='text-align: center; color: white; font-family: Source Code Pro; '>Where is this place?</h1>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: white; font-family: Source Code Pro; '>We can guess where your photo is taken at using our finetuned <a href=https://openai.com/blog/clip/>CLIP</a> model.</h5>", unsafe_allow_html=True)

st.markdown("""---""") 

column1, column2 = st.columns(2)
column1_forButton, column2_forButton = st.columns([2, 8.7])

class Image4Location():
    def __init__(self, model_type, city_csv, encode_text=False, save_text_features=False, use_finetuned_model=False) -> None:
        print("init")
        self.model_type = model_type
        if model_type == "vit":
            model_type_string = "ViT-L/14@336px"
        elif model_type == "resnet":
            model_type_string = "RN50x64"
        else:
            raise Exception('Invalid model_type parameter! Valid parameters : "vit", "resnet"')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_type_string, device=self.device)
        self.city_list = []
        self.generate_city_list(city_csv)
        if encode_text:
            self.text_features = self.encode_text_features(save_text_features)
        else:
            self.text_features = torch.load(f"text_features_{model_type}")
        if use_finetuned_model:
            self.use_finetuned_param()
        

    def use_finetuned_param(self):
        print("Loading finetuned model")
        checkpoint = torch.load("finetune.pt", map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def encode_text_features(self, save_text_features):
        text_features = []
        print("Encoding text features. This might take long.")
        with torch.no_grad():
            for x in range(self.text.shape[0]):
                text_features.append(self.model.encode_text(self.text[x].unsqueeze(0)))
        text_features = torch.cat(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        if save_text_features:
            torch.save(text_features, f"text_features_{self.model_type}")
        return text_features

    def generate_city_list(self, city_csv):
        cities = pandas.read_csv(city_csv)
        for city, country in zip(cities["city_ascii"].values, cities["country"].values):
            self.city_list.append(city + ", " + country)
        self.text = clip.tokenize(self.city_list).to(self.device)

    def predict(self, image_file):
        image = self.preprocess(Image.open(image_file)).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        return (values, indices, similarity, self.city_list)

def show_predictions(results_tuple):
    values = results_tuple[0]
    indices = results_tuple[1]
    similarity = results_tuple[2]
    city_list = results_tuple[3]
    with column2:
        st.markdown(
            "<h5 style='text-align: center; color: white; font-family: Source Code Pro; '>Top predictions:</h5>",
            unsafe_allow_html=True)
        for value, index in zip(values, indices):
            st.markdown("<h6 style='text-align: center; color: white; font-family: Source Code Pro; '>"f"{city_list[index]:>16s}: {100 * value.item():.2f}%""</h6>",
            unsafe_allow_html=True)

        values_2, indices_2 = similarity[0].topk(1)

        predicted_city = f"{city_list[indices_2]:>16s}".split(",")[0].strip(" ")
        predicted_country = f"{city_list[indices_2]:>16s}".split(",")[1]

        contents = urllib.request.urlopen(r"https://en.wikipedia.org/wiki/{}".format(predicted_city)).read()
        encoding = "utf-8"
        contents = contents.decode(encoding)
        soup = BeautifulSoup(contents, "html.parser")
        all_data = []
        for data in soup.find_all("p"):
            all_data.append(data.get_text())

        with st.expander("Click for more information"):
            st.write(' '.join(map(str, all_data))[:1000])

        geolocator = Nominatim(user_agent="GTA Lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(predicted_city + "," + predicted_country)

        lat = location.latitude
        long = location.longitude

        map_data = pd.DataFrame({'lat': [lat], 'lon': [long]})
        st.map(map_data, zoom=6)



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def create_model():
    model_type = "vit"
    city_csv = "citylist.csv"
    encode_text = False 
    save_text_features = False
    use_finetuned_model = True
    return Image4Location(model_type, city_csv, encode_text, save_text_features, use_finetuned_model)



if __name__ == "__main__":
    i4l = create_model()
    with column1:
        file_path = st.file_uploader("", type=["png", "jpg", "jpeg"], help="Press \"Browse files\" button to upload an image")

    if file_path is not None:
        image = Image.open(file_path)
        image = image.resize((700, 700))
        with column1:
            st.image(image)
            with column2_forButton:
                if st.button(label="Predict Location", help="Press this button to predict location of image"):
                    predict_tuple = i4l.predict(file_path)
                    show_predictions(predict_tuple)

    st.markdown("<h5 style='text-align: right; color: white; font-family: Source Code Pro; '>Arda Cihaner\nMutlu Cansever\nSadık Can Acar</h5>", unsafe_allow_html=True)
