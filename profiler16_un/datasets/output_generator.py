import xml.etree.cElementTree as ET


def generate_output(path, filename, author_id, doc_type, lang, age_group, gender):
    root = ET.Element("author")
    root.set('id', author_id)
    root.set('type', doc_type)
    root.set('lang', lang)
    root.set('age_group', age_group)
    root.set('gender', gender)
    tree = ET.ElementTree(root)
    tree.write("filename.xml")
