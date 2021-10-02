import unittest
from main import app


class BasicTests(unittest.TestCase):

    ############################
    ############################

    # executed prior to each test
    def setUp(self):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        self.app = app.test_client()

    # executed after each test
    def tearDown(self):
        pass

    def Classification(self, list):
        return self.app.post(
            '/Classification/',
            data=dict(list=list),
            follow_redirects=True)

    def Similarity(self, list):
        return self.app.post(
            '/Similarity/',
            data=dict(list=list),
            follow_redirects=True)

###############
###############

    def test_main_page(self):
        response = self.app.get('/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_home_page(self):
        response = self.app.get('/home/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_class_page(self):
        response = self.app.get('/Classification/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_sim_page(self):
        response = self.app.get('/Similarity/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_application_page(self):
        response = self.app.get('/application/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_class_form_page(self):
        response = self.Classification(list='ec:1.1.2.3')
        self.assertEqual(response.status_code, 200)

    def test_class_form_check1_page(self):
        response = self.Classification(list='JX')
        if b'Input not in the right format' in response.get_data():
            output = 1
        self.assertEqual(output, 1)

    def test_class_form_check2_page(self):
        response = self.Classification(list='ec:1.1.2.3')
        if b'The predicted class/classes:' in response.get_data():
            output = 1
        self.assertEqual(output, 1)

    def test_similarity_form_page(self):
        response = self.Similarity(list='ec:1.1.2.3')
        self.assertEqual(response.status_code, 200)

    def test_similarity_form_check1_page(self):
        response = self.Similarity(list='ec:88.1')
        if b'Input not in the right format' in response.get_data():
            output = 1
        self.assertEqual(output, 1)


if __name__ == "__main__":
    unittest.main()
