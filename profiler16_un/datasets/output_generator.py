import xml.etree.cElementTree as ET

def generate_output(path, filename, author_id, doc_type, lang, age_group, gender):
    root = ET.Element("author")
    doc = ET.SubElement(root, "doc")

    ET.SubElement(doc, "field1", name="blah").text = "some value1"
    ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"

    tree = ET.ElementTree(root)
    tree.write("filename.xml")
