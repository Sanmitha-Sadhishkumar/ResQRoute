from urllib.request import urlopen
from bs4 import BeautifulSoup
import os, requests

def download_image(url, folder_path):
    try:
        img_data = requests.get(url).content
        file_name = os.path.join(folder_path, url.split("/")[-1]) + '.jpg'
        with open(file_name, 'wb') as handler:
            handler.write(img_data)
        print(f"Downloaded: {file_name}")
    except Exception as e:
        print(f"Could not download {url}: {e}")

lpg = ['https://www.freepik.com/free-photos-vectors/lpg-gas',
        'https://www.istockphoto.com/search/2/image-film?phrase=lpg%20cylinder&mediatype=photography',
        'https://www.gettyimages.in/photos/lpg-cylinder',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=2',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=3',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=4',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=5',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=6',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=7',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=8',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=9',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=10',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=11',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=12',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=13',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=14',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=15',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=16',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=17',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=18',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=19',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=20',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=21',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=22',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=23',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=24',
        'https://www.gettyimages.in/photos/lpg-cylinder?page=25',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&phrase=lpg%20cylinder&page=2',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=3&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=4&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=5&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=6&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=7&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=8&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=9&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=10&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=11&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=12&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=13&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=14&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=15&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=16&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=17&phrase=lpg%20cylinder',
        'https://www.istockphoto.com/search/2/image-film?mediatype=photography&page=18&phrase=lpg%20cylinder',]

elderly_urls = ["https://www.gettyimages.in/search/2/image-film?phrase=indian%20elderly&sort=mostpopular&page=2",
        'https://www.gettyimages.in/search/2/image-film?page=3&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=4&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=5&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=6&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=7&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=8&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=9&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=10&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=11&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=12&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=13&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=14&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=15&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=16&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=17&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=18&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=19&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=20&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=21&phrase=indian%20elderly&sort=mostpopular',
        'https://www.gettyimages.in/search/2/image-film?page=22&phrase=indian%20elderly&sort=mostpopular',
        "https://unsplash.com/s/photos/elderly",
        "https://www.gettyimages.in/search/2/image-film?phrase=indian+elderly"]

child_urls = ['https://www.istockphoto.com/search/2/image-film?msockid=3877a4bd6ef66b9b38dcb5286af66546&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?phrase=children&page=2',
              'https://www.istockphoto.com/search/2/image-film?page=3&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=4&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=5&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=6&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=7&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=8&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=9&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=10&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=11&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=12&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=13&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=14&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=15&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=16&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=17&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=18&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=19&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=20&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=21&phrase=children',
              'https://www.istockphoto.com/search/2/image-film?page=22&phrase=children',
              ]

pictogram_urls = [
    'https://stock.adobe.com/in/search/images?k=chemical%20labelling',
    'https://www.complianceandrisks.com/topics/globally-harmonized-system/',
    'https://www.shutterstock.com/search/chemical-label',
    'https://www.istockphoto.com/search/2/image-film?phrase=chemical+label',
    'https://depositphotos.com/photos/chemical-label.html',
    'https://www.bradyindia.co.in/applications/ghs-labeling-requirements',
    'https://www.safetyhub.com/safety-training/introduction-to-ghs/',
    'https://stock.adobe.com/search/images?k=chemical+label',
    'https://teamdls.com/Label-Markets/Industrial-Labels/GHS-Chemical-Labels.htm',
    'https://www.kaggle.com/datasets/sagieppel/labpics-chemistry-labpics-medical/data',
    'https://www.coleparmer.in/p/ghs-flame-pictogram-labels/64906',
    'https://www.coleparmer.in/p/ghs-flame-over-circle-pictogram-labels/64907',
    'https://www.jjstech.com/ghs1054.html',
    'https://ehsdailyadvisor.blr.com/2018/04/ghs-pictogram-training-cheat-sheet/',
    'https://lawfilesext.leg.wa.gov/Law/WAC/WAC%20296%20%20TITLE/WAC%20296%20-901%20%20CHAPTER/WAC%20296%20-901%20-14026.htm',
    'https://www.shutterstock.com/search/flammalbe-image',
    'https://www.shutterstock.com/search/labelled-chemicals',
    'https://www.shutterstock.com/search/chermicals-in-bottles',
    'https://www.shutterstock.com/search/flame-over-circle',
    'https://www.shutterstock.com/search/flame-over-circle-symbol',
    'https://www.shutterstock.com/search/flame-over-circle-chemicals-symbol',
    'https://www.shutterstock.com/search/hazardous-substances-label',
    'https://www.shutterstock.com/search/labelled-hazardous',
    'https://www.chemicalindustryjournal.co.uk/back-to-the-basics-of-chemical-labelling',
    'https://in.vwr.com/store/product/3216634/vwr-labels-hazardous-substance-ghs-labels',
    'https://ohsonline.com/articles/2023/01/18/properly-store-and-label-hazardous-substances.aspx',
    'https://www.shutterstock.com/search/chemical-substance',
    'https://www.istockphoto.com/illustrations/chemical-label',
    'https://www.istockphoto.com/search/2/image-film?mediatype=illustration&phrase=hazardous%20chemicals',
    'https://www.istockphoto.com/search/2/image-film?mediatype=illustration&phrase=hazardous%20chemicals%20label&servicecontext=srp-related'
    'https://www.istockphoto.com/search/2/image-film?mediatype=illustration&phrase=hazardous%20chemicals%20',
    'https://www.istockphoto.com/search/2/image-film?mediatype=illustration&phrase=hazardous%20chemicals%20construction&servicecontext=srp-related',
    'https://www.istockphoto.com/search/2/image-film?mediatype=illustration&phrase=hazardous%20chemicals%20storage&servicecontext=srp-related',
    'https://www.hague-group.com/chemical-hazard-labelling-our-comprehensive-guide/',
    'https://www.herma.com/label/products/labels-for-hazardous-substances-and-dangerous-goods/hazardous-substance-labels/',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20substance%20label&sort=mostpopular',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=chemical%20labels&sort=mostpopular',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&sort=mostpopular',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=2',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=3',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=4',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=5',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=6',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=7',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=8',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20chemical%20labels&suppressfamilycorrection=true&sort=mostpopular&page=9',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=hazardous%20labels&sort=mostpopular',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=chemical%20labels&sort=mostpopular',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=2',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=3',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=4',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=5',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=6',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=7',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=8',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=2',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=3',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=4',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=5',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=6',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=7',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=8',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=9',
    'https://www.gettyimages.in/search/2/image-film?family=creative&phrase=flammable%20labels&sort=mostpopular&page=10'
]


def web_scrap(urls, folder_path):
  os.makedirs(folder_path, exist_ok=True)

  for i in urls:
    print(f'URL : {i}')
    htmldata = urlopen(i)
    soup = BeautifulSoup(htmldata, 'html.parser')
    images = soup.find_all('img')

    for item in images:
      print(item['src'])
      img_url = item['src']
      download_image(img_url, folder_path)

  files = os.listdir(folder_path)
  object_crop_files = [file for file in files if file.endswith('.jpg')]
  return 'Scraped images count : '+ str(len(object_crop_files))
