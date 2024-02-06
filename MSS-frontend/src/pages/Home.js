import React from 'react';
import Footer from '../components/Footer';
import Hero from '../components/Hero';
import Portfolio from '../components/Portfolio';
// import Services from '../components/Services';
import MTable from '../components/MTable';


const Home = () => {
    return (
        <>
            <Hero />
            {/* <Services /> */}
            < MTable />
            <Portfolio />
            <Footer />
        </>

    )
}

export default Home;

